from learning.modules import *
from learning.handler import *
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime
import time as t

### ADDING PAHO-MQTT LINES ###
import paho.mqtt.client as mqtt

def on_connect(mqttc, obj, flags, reason_code, properties):
    print("reason_code: " + str(reason_code))


def on_message(mqttc, obj, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))


def on_publish(mqttc, obj, mid, reason_code, properties):
    print("mid: " + str(mid))


def on_log(mqttc, obj, level, string):
    print(string)

# If you want to use a specific client id, use
# mqttc = mqtt.Client("client-id")
# but note that the client id must be unique on the broker. Leaving the client
# id parameter empty will generate a random id for you.
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
# Uncomment to enable debug messages
# mqttc.on_log = on_log

### RL LINES ###
def run(train=True,model_name=None,epochs=1,steps=600,gamma=0.8,epsilon=0.3,option_rules='paper1',observation='tf',point_reward='wt'):
    # variabe training
    epochs = epochs
    steps = steps
    if (option_rules == 'paper1'):
        rules = paper1_rules(observation, point_reward)
    else:
        print('Rules not found')
        sys.exit()

    # variable report
    report = list()
    report_allred = list()

    # starting brain
    start_sumo_cmd()
    all_junctions = traci.trafficlight.getIDList()
    total_junctions = list(range(len(all_junctions)))
    stop_sumo()

    # model untuk training, harus ada nama parameter brain nya
    if model_name is not None :
        brain = Agent(
            gamma=gamma,
            epsilon=epsilon, #nilai peluang pembuat keputusan
            lr=0.1, #lerning rate
            input_dims=rules.max_lanes * rules.observasi_len, #sesuaikan dengan jumlah kaki simpang
            fc1_dims=256, #Jumlah neuron dalam lapisan fully connected pertama (hidden layer 1)
            fc2_dims=256, #Jumlah neuron dalam lapisan fully connected pertama (hidden layer 2)
            batch_size=1024, #Ukuran batch yang digunakan saat melakukan pembelajaran (training)
            n_actions=len(rules.action_list), #aksi yang perlu diambil
            junctions=total_junctions, #jumlah trafic light
        )

    # kalau mode testing
    if not train and model_name is not None:
        brain.Q_eval.load_state_dict(torch.load(f'result/{model_name}.bin',map_location=brain.Q_eval.device))

    # variable learning
    best_wt = np.inf
    total_wt = list()
    epoch_wt = 0

    trafic_light = dict()
    agent_log = dict()
    simulation_log = dict()

    # iterasi learning
    for e in range(epochs):
        # run traci sumo
        if train: start_sumo_cmd()
        # else: start_sumo_gui()
        else: start_sumo_cmd()

        # Connect MQTT sebelum start & publish
        mqttc.connect("raspberrypi.local", 1883, 60) ## Opsi awal 'mqtt.eclipseprojects.io'

        # init all data traficlight
        for junction_number, junction in enumerate(all_junctions):
            trafic_light[junction] = Trafic_light(junction, rules.duration_all_red, rules.duration_yellow_red, rules.duration_max_phase)
            simulation_log[junction] = dict()
            agent_log[junction] = dict()
            agent_log[junction]['prev_action'] = 0
            agent_log[junction]['prev_observation'] = [0] * trafic_light[junction].lanes_number * rules.observasi_len


        runtime = 0
        epoch_wt = 0

        _change_all_red = list()
        _change_yell_red = list()
        _change_red_green = list()

        _option_light_status = list()
        _option_green_times = list()


        # start run simulation every seconds
        while runtime <= steps:
            traci.simulationStep()

            # periksa semua kondisi simpang/junction untuk perhitungan agent
            for junction_number, junction in enumerate(all_junctions):
                all_lanes = traci.trafficlight.getControlledLanes(junction)

                # TF : Trafic Flow
                # WT : Witing Time
                # AS : Average Speed
                # AWT : Average Waiting Time
                simulation_log[junction]['e']  = e
                simulation_log[junction]['s']  = runtime
                simulation_log[junction]['cd'] = trafic_light[junction].curr_duration_phase
                simulation_log[junction]['tf'] = trafic_light[junction].totalVehiclePerLane()
                simulation_log[junction]['wt'] = trafic_light[junction].totalWaitingTimePerlane()
                simulation_log[junction]['as'] = trafic_light[junction].avgSpeedPerLane()
                simulation_log[junction]['light'] = trafic_light[junction].statusLight()
                starttime = t.time()
                
                report.append(copy.deepcopy(simulation_log))
                epoch_wt += sum(list(simulation_log[junction]['wt'].values()))

                with open('data.json', 'w') as f:
                    json.dump(simulation_log, f)                

                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Epoch {epochs} | Obs {observation} | Gamma {gamma} | Epsilon {epsilon}")
                print(json.dumps(simulation_log, indent=4))
                
                print("Start time: " + str(starttime))

                ## MQTT Message line ##
                #starttime = t.time()
                light_message = simulation_log[junction]['light']
                mqttc.loop_start()
                mqttc.publish("tl/lights", light_message, qos=2)
                mqttc.publish("tl/starttime", starttime, qos=2)

                # if model_name is not None :
                #     print('brain')
                # else:

                if trafic_light[junction].curr_duration_phase == 0:
                    report_allred.append(copy.deepcopy(simulation_log))
                    # start set control auto RL
                    if model_name is not None :
                        #memberikan reward berdasarkan kondisi saat ini
                        reward = rules.countReward(junction, simulation_log) 
                        # mengambil matriks jumlah kendaraan untuk simpang saat ini
                        CURR_MATRIX_VEHICLE = rules.observationMatix(junction, simulation_log)
                        # mengambil matriks jumlah kendaraan sebelumnya
                        PREV_MATRIX_VEHICLE = agent_log[junction]['prev_observation']
                        # mengganti nilai matriks sebelumnya dengan nilai matriks saat ini
                        agent_log[junction]['prev_observation'] = CURR_MATRIX_VEHICLE
                        # menimpan nilai transisi dalam brain agent
                        brain.store_transition(PREV_MATRIX_VEHICLE, CURR_MATRIX_VEHICLE, agent_log[junction]['prev_action'], reward, (runtime==steps),junction_number)

                        #selecting new action based on current state
                        action_agent = brain.choose_action(CURR_MATRIX_VEHICLE)
                        time_green_from_action = rules.action_list[action_agent]

                        _change_all_red = []
                        _change_yell_red = []
                        _change_red_green = []

                        _option_light_status = []
                        _option_green_times = []

                        # simpan action saat ini
                        agent_log[junction]['prev_action'] = action_agent

                        # set phase controll
                        if time_green_from_action[0] != 0:
                            _option_light_status.append(['rr', 'yr', 'Gr'])
                            _option_green_times.append(time_green_from_action[0])
                        if time_green_from_action[1] != 0:
                            _option_light_status.append(['rr', 'ry', 'rG'])
                            _option_green_times.append(time_green_from_action[1])


                        # set timing trafic all red
                        objectTL = trafic_light[junction]

                        # set timing trafic all red
                        if objectTL.duration_all_red != 0:
                            _change_all_red.append(0)
                            if time_green_from_action[0] != 0 and time_green_from_action[1] != 0:
                                objectTL.duration_all_red + objectTL.duration_yellow_red + objectTL.duration_green_red_fixed
                                _change_all_red.append(objectTL.duration_all_red + objectTL.duration_yellow_red + time_green_from_action[0])
                        # set timing trafic yellow red
                        if objectTL.duration_yellow_red != 0:
                            _change_yell_red.append(objectTL.duration_all_red ) 
                            if time_green_from_action[0] != 0 and time_green_from_action[1] != 0:
                                _change_yell_red.append(objectTL.duration_all_red + objectTL.duration_yellow_red + time_green_from_action[0] + objectTL.duration_all_red ) 
                        # set timing trafic yellow red
                        if objectTL.duration_green_red_fixed != 0:
                            _change_red_green.append(objectTL.duration_all_red + objectTL.duration_yellow_red ) 
                            if time_green_from_action[0] != 0 and time_green_from_action[1] != 0:
                                _change_red_green.append(objectTL.duration_all_red + objectTL.duration_yellow_red + time_green_from_action[0] + objectTL.duration_all_red + objectTL.duration_yellow_red ) 


                        # set total waktu phase ini
                        total_duration_phase = 0
                        for time in time_green_from_action:
                            if time != 0:
                                total_duration_phase += (objectTL.duration_all_red + objectTL.duration_yellow_red + time)

                        trafic_light[junction].total_duration_phase = total_duration_phase

                        if train:
                            brain.learn(junction_number)
                    # end set control auto RL
                    
                    # start set control manual
                    else:
                        trafic_light[junction].setPhaseTimeBased()

                        _option_light_status = [
                            ['rr', 'yr', 'Gr'],
                            ['rr', 'ry', 'rG']
                        ]
                        _option_green_times = [
                            trafic_light[junction].duration_green_red_fixed,
                            trafic_light[junction].duration_green_red_fixed
                        ]

                        # set timing trafic all red
                        _change_all_red = trafic_light[junction].getTimeChangeAllRedFixed()
                        _change_yell_red = trafic_light[junction].getTimeChangeYellRedFixed()
                        _change_red_green = trafic_light[junction].getTimeChangeGreenRedFixed()
                    # end set control manual


                    if (len(_change_all_red) > 0): trafic_light[junction].phaseDuration(trafic_light[junction].duration_all_red, 'rr')
                    elif (len(_change_yell_red) > 0): trafic_light[junction].phaseDuration(trafic_light[junction].duration_all_red, _option_light_status[0][1])
                    elif (len(_change_red_green) > 0): trafic_light[junction].phaseDuration(trafic_light[junction].duration_all_red, _option_light_status[0][2])

                    trafic_light[junction].curr_duration_phase += 1
                elif trafic_light[junction].curr_duration_phase in _change_all_red:
                    index_phase = _change_all_red.index(trafic_light[junction].curr_duration_phase)
                    trafic_light[junction].phaseDuration(trafic_light[junction].duration_all_red, _option_light_status[index_phase][0]) # rr
                    trafic_light[junction].curr_duration_phase += 1

                elif trafic_light[junction].curr_duration_phase in _change_yell_red:
                    index_phase = _change_yell_red.index(trafic_light[junction].curr_duration_phase)
                    trafic_light[junction].phaseDuration(trafic_light[junction].duration_yellow_red, _option_light_status[index_phase][1]) # yr / ry
                    trafic_light[junction].curr_duration_phase += 1

                elif trafic_light[junction].curr_duration_phase in _change_red_green:
                    index_phase = _change_red_green.index(trafic_light[junction].curr_duration_phase)
                    trafic_light[junction].phaseDuration(_option_green_times[index_phase], _option_light_status[index_phase][2]) # rr
                    trafic_light[junction].curr_duration_phase += 1

                elif trafic_light[junction].curr_duration_phase == trafic_light[junction].total_duration_phase-1:
                    trafic_light[junction].curr_duration_phase = 0

                else:
                    trafic_light[junction].curr_duration_phase += 1

            runtime += 1
            t.sleep(1) # add delay for the output loop
        # end run simulation

        total_wt.append(epoch_wt)
        if epoch_wt < best_wt:
            best_wt = epoch_wt
            if train and model_name is not None:
                brain.save(model_name)

        # end traci sumo
        stop_sumo()

        light_message = "end"
        mqttc.publish("tl/lights", light_message, qos=2)
        mqttc.loop_stop()

        mqttc.disconnect()

    curr = datetime.datetime.now()
    curr = curr.strftime("%Y-%m-%d %H_%M")
    mode = 'training' if train else 'testing'
    model_name = model_name if model_name is not None else 'timingbase'

    # plot
    plt.plot(list(range(len(total_wt))),total_wt)
    plt.xlabel("epochs")
    plt.ylabel("total waiting time")
    plt.savefig(f'plot/{option_rules}_{mode}_{model_name}_e{epsilon}_g{gamma}_{curr}.png')
    # plt.show()

    with open(f'log/{option_rules}_{mode}_{model_name}_e{epsilon}_g{gamma}_{curr}.json', 'w') as json_file:
        json.dump(report, json_file, indent=4)
    # with open(f'log/{mode}_{option_rules}_{model_name}_{curr}_onaction.json', 'w') as json_file:
    #     json.dump(report_allred, json_file, indent=4)

### PYQT5 LINES ###
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 450)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, -1)
        self.verticalLayout.setSpacing(4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_input_data = QtWidgets.QLabel(self.centralwidget)
        self.label_input_data.setObjectName("label_input_data")
        self.verticalLayout.addWidget(self.label_input_data)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setStyleSheet("background-color: rgb(249, 249, 249);")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 186, 409))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_train = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_train.setObjectName("label_train")
        self.verticalLayout_4.addWidget(self.label_train)
        self.input_train = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.input_train.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.input_train.setObjectName("input_train")
        self.input_train.addItem("")
        self.input_train.addItem("")
        self.verticalLayout_4.addWidget(self.input_train)
        self.label_model_name = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_model_name.setObjectName("label_model_name")
        self.verticalLayout_4.addWidget(self.label_model_name)
        self.input_model_name = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.input_model_name.sizePolicy().hasHeightForWidth())
        self.input_model_name.setSizePolicy(sizePolicy)
        self.input_model_name.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.input_model_name.setInputMethodHints(QtCore.Qt.ImhLowercaseOnly)
        self.input_model_name.setInputMask("")
        self.input_model_name.setObjectName("input_model_name")
        self.verticalLayout_4.addWidget(self.input_model_name)
        self.label_epochs = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_epochs.setObjectName("label_epochs")
        self.verticalLayout_4.addWidget(self.label_epochs)
        self.input_epochs = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.input_epochs.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.input_epochs.setProperty("value", 1)
        self.input_epochs.setObjectName("input_epochs")
        self.verticalLayout_4.addWidget(self.input_epochs)
        self.label_steps = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_steps.setObjectName("label_steps")
        self.verticalLayout_4.addWidget(self.label_steps)
        self.input_steps = QtWidgets.QSpinBox(self.scrollAreaWidgetContents)
        self.input_steps.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.input_steps.setMaximum(32768)
        self.input_steps.setProperty("value", 1800)
        self.input_steps.setObjectName("input_steps")
        self.verticalLayout_4.addWidget(self.input_steps)
        self.label_gamma = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_gamma.setObjectName("label_gamma")
        self.verticalLayout_4.addWidget(self.label_gamma)
        self.input_gamma = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents)
        self.input_gamma.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.input_gamma.setProperty("value", 0.8)
        self.input_gamma.setObjectName("input_gamma")
        self.verticalLayout_4.addWidget(self.input_gamma)
        self.label_epsilon = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_epsilon.setObjectName("label_epsilon")
        self.verticalLayout_4.addWidget(self.label_epsilon)
        self.input_epsilon = QtWidgets.QDoubleSpinBox(self.scrollAreaWidgetContents)
        self.input_epsilon.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.input_epsilon.setProperty("value", 0.3)
        self.input_epsilon.setObjectName("input_epsilon")
        self.verticalLayout_4.addWidget(self.input_epsilon)
        self.label_rules = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_rules.setObjectName("label_rules")
        self.verticalLayout_4.addWidget(self.label_rules)
        self.input_rules = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.input_rules.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.input_rules.setObjectName("input_rules")
        self.input_rules.addItem("")
        self.verticalLayout_4.addWidget(self.input_rules)
        self.label_observation = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_observation.setObjectName("label_observation")
        self.verticalLayout_4.addWidget(self.label_observation)
        self.input_observation = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.input_observation.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.input_observation.setObjectName("input_observation")
        self.input_observation.addItem("")
        self.input_observation.addItem("")
        self.input_observation.addItem("")
        self.verticalLayout_4.addWidget(self.input_observation)
        self.label_point_reward = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_point_reward.setObjectName("label_point_reward")
        self.verticalLayout_4.addWidget(self.label_point_reward)
        self.input_point_reward = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.input_point_reward.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.input_point_reward.setObjectName("input_point_reward")
        self.input_point_reward.addItem("")
        self.input_point_reward.addItem("")
        self.input_point_reward.addItem("")
        self.verticalLayout_4.addWidget(self.input_point_reward)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        self.button_reset = QtWidgets.QPushButton(self.centralwidget)
        self.button_reset.setStyleSheet("background-color: #ffffff; color: #f02849;")
        self.button_reset.setObjectName("button_reset")
        self.verticalLayout.addWidget(self.button_reset)
        self.button_start = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_start.sizePolicy().hasHeightForWidth())
        self.button_start.setSizePolicy(sizePolicy)
        self.button_start.setStyleSheet("background-color: rgb(85, 170, 255); color: rgb(255, 255, 255); \n"
"font: 75 8pt \"MS Shell Dlg 2\";")
        self.button_start.setObjectName("button_start")
        self.verticalLayout.addWidget(self.button_start)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_traffic = QtWidgets.QLabel(self.centralwidget)
        self.label_traffic.setObjectName("label_traffic")
        self.verticalLayout_2.addWidget(self.label_traffic)
        self.output_traffic = QtWidgets.QTextBrowser(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_traffic.sizePolicy().hasHeightForWidth())
        self.output_traffic.setSizePolicy(sizePolicy)
        self.output_traffic.setObjectName("output_traffic")
        self.verticalLayout_2.addWidget(self.output_traffic)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_log = QtWidgets.QLabel(self.centralwidget)
        self.label_log.setObjectName("label_log")
        self.verticalLayout_3.addWidget(self.label_log)
        self.output_log = QtWidgets.QTextBrowser(self.centralwidget)
        self.output_log.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_log.sizePolicy().hasHeightForWidth())
        self.output_log.setSizePolicy(sizePolicy)
        self.output_log.setObjectName("output_log")
        self.verticalLayout_3.addWidget(self.output_log)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionRestart = QtWidgets.QAction(MainWindow)
        self.actionRestart.setObjectName("actionRestart")
        self.actionDocumentation = QtWidgets.QAction(MainWindow)
        self.actionDocumentation.setObjectName("actionDocumentation")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menuFile.addAction(self.actionClose)
        self.menuHelp.addAction(self.actionDocumentation)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.label_train.setBuddy(self.input_train)
        self.label_model_name.setBuddy(self.input_model_name)
        self.label_epochs.setBuddy(self.input_epochs)
        self.label_steps.setBuddy(self.input_steps)
        self.label_gamma.setBuddy(self.input_gamma)
        self.label_epsilon.setBuddy(self.input_epsilon)
        self.label_rules.setBuddy(self.input_rules)
        self.label_observation.setBuddy(self.input_observation)
        self.label_point_reward.setBuddy(self.input_point_reward)

        self.retranslateUi(MainWindow)
        self.input_train.setCurrentIndex(1)
        self.input_observation.setCurrentIndex(0)
        self.input_point_reward.setCurrentIndex(2)
        self.actionClose.triggered.connect(MainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.button_start.clicked.connect(self.getInfo)
        self.button_reset.clicked.connect(self.resetInfo) # type: ignore

        self.output_log.append("[{0}] App is opened.".format(datetime.now()))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RL for Two Lane Traffic Light Simulator"))
        self.label_input_data.setText(_translate("MainWindow", "Input Data"))
        self.label_train.setText(_translate("MainWindow", "Training Mode"))
        self.input_train.setItemText(0, _translate("MainWindow", "Enabled"))
        self.input_train.setItemText(1, _translate("MainWindow", "Disabled"))
        self.label_model_name.setText(_translate("MainWindow", "Model Name"))
        self.input_model_name.setPlaceholderText(_translate("MainWindow", "Default: None"))
        self.label_epochs.setText(_translate("MainWindow", "Epochs Number"))
        self.label_steps.setText(_translate("MainWindow", "Steps Number"))
        self.label_gamma.setText(_translate("MainWindow", "Gamma"))
        self.label_epsilon.setText(_translate("MainWindow", "Epsilon"))
        self.label_rules.setText(_translate("MainWindow", "Rules Option"))
        self.input_rules.setItemText(0, _translate("MainWindow", "paper1"))
        self.label_observation.setText(_translate("MainWindow", "Data Observation"))
        self.input_observation.setItemText(0, _translate("MainWindow", "TF - Traffic Flow"))
        self.input_observation.setItemText(1, _translate("MainWindow", "AS - Average Speed"))
        self.input_observation.setItemText(2, _translate("MainWindow", "WT - Waiting Time"))
        self.label_point_reward.setText(_translate("MainWindow", "Point Reward"))
        self.input_point_reward.setItemText(0, _translate("MainWindow", "TF - Traffic Flow"))
        self.input_point_reward.setItemText(1, _translate("MainWindow", "AS - Average Speed"))
        self.input_point_reward.setItemText(2, _translate("MainWindow", "WT - Waiting Time"))
        self.button_reset.setText(_translate("MainWindow", "Reset Data"))
        self.button_start.setText(_translate("MainWindow", "Start"))
        self.label_traffic.setText(_translate("MainWindow", "Traffic Conditions"))
        self.label_log.setText(_translate("MainWindow", "Console Logs"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionClose.setText(_translate("MainWindow", "Exit"))
        self.actionRestart.setText(_translate("MainWindow", "Restart"))
        self.actionDocumentation.setText(_translate("MainWindow", "Documentation"))
        self.actionAbout.setText(_translate("MainWindow", "About"))

    def getInfo(self):
        global train, model_name, epochs, steps, gamma, epsilon, rules, observation, point_reward
        print("Training Mode: {0}".format(self.input_train.currentText()))
        if self.input_train.currentText() == "Enabled":
            train = True
        print("Model Name: {0}".format(self.input_model_name.text()))
        model_name = self.input_model_name.text()
        print("Epochs Number: {0}".format(self.input_epochs.value()))
        epochs = self.input_epochs.value()
        print("Steps Number: {0}".format(self.input_steps.value()))
        steps = self.input_steps.value()
        print("Gamma: {0}".format(self.input_gamma.value()))
        gamma = self.input_gamma.value()
        print("Epsilon: {0}".format(self.input_epsilon.value()))
        epsilon = self.input_epsilon.value()
        print("Rules: {0}".format(self.input_rules.currentText()))
        rules = self.input_rules.currentText()
        print("Observation: {0}".format(self.input_observation.currentText()))
        observation_index = self.input_observation.currentIndex()
        if observation_index == 0:
            observation = 'TF'
        elif observation_index == 1:
            observation = 'AS'
        else: observation = 'WT'
        print("Reward: {0}".format(self.input_point_reward.currentText()))
        reward_index = self.input_point_reward.currentIndex()
        if reward_index == 0:
            point_reward = 'TF'
        elif reward_index == 1:
            point_reward = 'AS'
        else: point_reward = 'WT'
        self.output_log.append("[{0}] The program is running!".format(datetime.now()))
        run(train=train,model_name=model_name,epochs=epochs,steps=steps,
            gamma=gamma,epsilon=epsilon,option_rules=rules,
            observation=observation,point_reward=point_reward)

    def resetInfo(self):
        self.input_train.setCurrentIndex(1)
        self.input_model_name.clear()
        self.input_epochs.setValue(1)
        self.input_steps.setValue(1800)
        self.input_gamma.setValue(0.80)
        self.input_epsilon.setValue(0.3)
        self.input_rules.setCurrentIndex(0)
        self.input_observation.setCurrentIndex(0)
        self.input_point_reward.setCurrentIndex(2)

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    train = options.train
    epochs = options.epochs
    steps = options.steps
    rules = options.rules.lower()
    observation = options.observation.lower()
    point_reward = options.point_reward.lower()
    model_name = options.model_name
    gamma = options.gamma
    epsilon = options.epsilon
    print(options)

    if rules not in ['paper1', 'paper2', 'paper3']:
        print(f"option rules {rules} not available. check option with --help")
    if observation not in ['WT', 'AS', 'TF']:
        print(f"option observation {observation} not available. check option with --help")
    if point_reward not in ['WT', 'AS', 'TF']:
        print(f"option point reward {point_reward} not available. check option with --help")
        
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

    # run(train=train,model_name=model_name,epochs=epochs,steps=steps,gamma=gamma,epsilon=epsilon,option_rules=rules,observation=observation,point_reward=point_reward)