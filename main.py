from learning.modules import *
from learning.handler import *
from time import sleep

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
        mqttc.connect("broker.emqx.io", 1883, 60) ## Opsi awal 'mqtt.eclipseprojects.io'

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
                report.append(copy.deepcopy(simulation_log))
                epoch_wt += sum(list(simulation_log[junction]['wt'].values()))

                ## MQTT Message line ##
                light_message = simulation_log[junction]['light']
                mqttc.loop_start()
                mqttc.publish("tl/lights", light_message, qos=2)
                

                with open('data.json', 'w') as f:
                    json.dump(simulation_log, f)                

                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Epoch {epochs} | Obs {observation} | Gamma {gamma} | Epsilon {epsilon}")
                print(json.dumps(simulation_log, indent=4))

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
            sleep(1) # add delay for the output loop
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

    run(train=train,model_name=model_name,epochs=epochs,steps=steps,gamma=gamma,epsilon=epsilon,option_rules=rules,observation=observation,point_reward=point_reward)
