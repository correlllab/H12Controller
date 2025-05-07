import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
TOPIC_HIGHSTATE = 'rt/sportmodestate'
TOPIC_HANDSTATE = 'rt/inspire/state'
NUM_MOTOR = 27

class StateSubscriber:
    def __init__(self):
        # variable tracking states
        self._q = np.zeros(NUM_MOTOR)
        self._dq = np.zeros(NUM_MOTOR)
        self._tau = np.zeros(NUM_MOTOR)

        # initialize channel
        ChannelFactoryInitialize(id=0)
        # subscribe low state
        self.low_state_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.low_state_subscriber.Init(self.subscribe_low_state, 10)

    def subscribe_low_state(self, msg: LowState_):
        for i in range(NUM_MOTOR):
            self._q[i] = msg.motor_state[i].q
            self._dq[i] = msg.motor_state[i].dq
            self._tau[i] = msg.motor_state[i].tau_est

    @property
    def q(self):
        return np.copy(self._q)

    @property
    def dq(self):
        return np.copy(self._dq)

    @property
    def get_tau(self):
        return np.copy(self._tau)
