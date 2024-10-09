# RL-TraficLight-TwoLane

- Discount Factor : [**0, 0,2, 0,4, 0,6, 0,8, 1**]
- Epoch : [**50**, 100, 500, 1000]
- Epsilon : [**0, 0.3, 0.6, 0.9**]
- Durasi simulasi 1800s,

untuk proses training : i = epsilon
python main.py --train -e {50} -m {AS}vs{WT}-1800x{50}\_e{0.0}\_g{0.0} -r paper1 -o {AS} -p {WT} -s 1800 -g {0.0} -i {0.0} &

untuk proses testing :
python main.py -e {50} -m {AS}vs{WT}-1800x{50}\_e{0.0}\_g{0.0} -r paper1 -o {AS} -p {WT} -s 1800 -g {0.0} -i {0.0} &

Yoanda AS
