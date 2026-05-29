def gripper_q_robomanip_to_maniskill(q_robomanip: float) -> float:
    return (q_robomanip - 850.0) / (-1000.0)


def gripper_q_maniskill_to_robomanip(q_maniskill: float) -> float:
    return q_maniskill * (-1000.0) + 850.0
