def gripper_q_robomanip_to_maniskill(q_robomanip: float) -> float:
    """Convert RoboManip gripper position scalar to ManiSkill scale."""

    return (q_robomanip - 850.0) / (-1000.0)


def gripper_qvel_robomanip_to_maniskill(qvel_robomanip: float) -> float:
    """Convert RoboManip gripper velocity scalar to ManiSkill scale."""

    return qvel_robomanip / (-1000.0)


def gripper_q_maniskill_to_robomanip(q_maniskill: float) -> float:
    """Convert ManiSkill gripper position scalar to RoboManip scale."""

    return q_maniskill * (-1000.0) + 850.0

if __name__=="__main__":
    print(gripper_q_robomanip_to_maniskill(119.0))