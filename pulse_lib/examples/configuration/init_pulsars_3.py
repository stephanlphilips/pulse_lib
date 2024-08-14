import qcodes as qc

from qblox_instruments import Cluster, ClusterType

try:
    from q1simulator import Cluster as SimCluster
    _q1simulator_found = True
except:
    print('package q1simulator not found')
    _q1simulator_found = False


if not qc.Station.default:
    station = qc.Station()
else:
    station = qc.Station.default


_use_simulator = True
_use_dummy = False


try:
    cluster = station['Qblox_Cluster']
    cluster.reset()
    qcm0 = cluster.module8
    qrm1 = cluster.module10
    qcm2 = cluster.module6
except:
    if _use_simulator:
        cfg = {
            6: "QCM",
            8: "QCM",
            10: "QRM",
            }
        cluster =  SimCluster('Q1Simulator_Cluster', modules=cfg)
    elif _use_dummy:
        cfg = {
            6:ClusterType.CLUSTER_QCM,
            8:ClusterType.CLUSTER_QCM,
            10:ClusterType.CLUSTER_QRM
            }
        cluster = Cluster('Qblox_Cluster', '192.168.0.2', dummy_cfg=cfg)
        # set property is_dummy to use in Q1Pulse state checking
        cluster.is_dummy = True
    else:
        cluster = Cluster('Qblox_Cluster', '192.168.0.2')

    station.add_component(cluster)
    cluster.reset()

    print('Cluster:')
    print(cluster.get_system_state())
    for module in cluster.modules:
        if module.present():
            rf = '-RF' if module.is_rf_type else ''
            print(f'  slot {module.slot_idx}: {module.module_type}{rf}')

    qcm0 = cluster.module8
    qrm1 = cluster.module10
    qcm2 = cluster.module6
    station.add_component(qcm0)
    station.add_component(qrm1)
    station.add_component(qcm2)
