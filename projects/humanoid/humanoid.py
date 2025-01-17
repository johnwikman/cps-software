from helper_functions import DynamixelHandler
import time

NECK_Y = 1
NECK_X = 2
NECK_Z = 3
R_SHOULDER_Y = 4
R_SHOULDER_X = 5
R_SHOULDER_Z = 6
R_ELBOW = 7
L_SHOULDER_Y = 8
L_SHOULDER_X = 9
L_SHOULDER_Z = 10
L_ELBOW = 11
TORSO_Z = 12
TORSO_Y = 13
TORSO_X = 14
R_HIP_X = 15
R_HIP_Z = 16
R_HIP_Y = 17
R_KNEE = 18
R_FOOT = 19
L_HIP_X = 20
L_HIP_Z = 21
L_HIP_Y = 22
L_KNEE = 23
L_FOOT = 24

class Humanoid:
    def __init__(self):
        self.dynamixel_handler = DynamixelHandler()
        self.all_ids = list(range(1, 25))

    def read_all_servos(self):
        return self.dynamixel_handler.read_servo_positions(self.all_ids)
    
    def disable_torques(self):
        self.dynamixel_handler.disable_torques(self.all_ids)

    def go_to_base_position(self):
        base_positions = [2048]*24
        durations = [5000]*24
        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions, durations)
        time.sleep(5)

    def stand(self):
        base_positions = [2048]*25
        base_positions[R_SHOULDER_X] = 2850          # Right shoulder
        base_positions[L_SHOULDER_X] = 1250          # Left shoulder
        base_positions[R_ELBOW] = 2400          # Right elbow
        base_positions[L_ELBOW] = 1850         # Left elbow

        first_base_positions = base_positions.copy()

        durations = [4000]*24
        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions[1:], durations)
        time.sleep(10)

        base_positions[NECK_Y] = 1773
        base_positions[NECK_X] = 2153
        base_positions[NECK_Z] = 1459

        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions[1:], durations)
        time.sleep(4)

        base_positions[NECK_Y] = 1881
        base_positions[NECK_X] = 2106
        base_positions[NECK_Z] = 2675

        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions[1:], durations)
        time.sleep(4)

        base_positions[NECK_Y] = 2230
        base_positions[NECK_X] = 2062
        base_positions[NECK_Z] = 2089
        base_positions[L_SHOULDER_Y] = 1500
        base_positions[R_SHOULDER_Y] = 2500
        base_positions[L_SHOULDER_Z] = 2300
        base_positions[R_SHOULDER_Z] = 1700
        base_positions[L_ELBOW] = 1500
        base_positions[R_ELBOW] = 2500
        base_positions[TORSO_Y] = 1970

        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions[1:], durations)
        time.sleep(4)

        self.dynamixel_handler.move_many_servos(self.all_ids, first_base_positions[1:], durations)
        time.sleep(4)

    def sit(self):
        sit_pos = [2048]*25
        sit_pos[R_SHOULDER_X] = 2850          # Right shoulder
        sit_pos[L_SHOULDER_X] = 1250          # Left shoulder
        sit_pos[R_ELBOW] = 2400          # Right elbow
        sit_pos[L_ELBOW] = 1850         # Left elbow

        sit_pos[R_HIP_X] = 1961
        sit_pos[R_HIP_Z] = 2098
        sit_pos[R_HIP_Y] = 961
        sit_pos[R_KNEE] = 3077
        sit_pos[R_FOOT] = 2669
        sit_pos[L_HIP_X] = 2060
        sit_pos[L_HIP_Z] = 1893
        sit_pos[L_HIP_Y] = 3149
        sit_pos[L_KNEE] = 1093
        sit_pos[L_FOOT] = 1265

        sit_pos = sit_pos[1:]

        durations = [5000]*24

        self.dynamixel_handler.move_many_servos(self.all_ids, sit_pos, durations)
        time.sleep(6)

    def side_split(self):
        side_split_pos = [2048]*25
        side_split_pos[R_SHOULDER_Y] = 2296
        side_split_pos[R_SHOULDER_X] = 2738
        side_split_pos[R_SHOULDER_Z] = 2241
        side_split_pos[R_ELBOW] = 2565
        side_split_pos[L_SHOULDER_Y] = 1893
        side_split_pos[L_SHOULDER_X] = 1473
        side_split_pos[L_SHOULDER_Z] = 1651
        side_split_pos[L_ELBOW] = 1492
        side_split_pos[TORSO_Z] = 2127
        side_split_pos[TORSO_Y] = 1985
        side_split_pos[TORSO_X] = 2096
        side_split_pos[R_HIP_X] = 2077
        side_split_pos[R_HIP_Z] = 2970
        side_split_pos[R_HIP_Y] = 1050
        side_split_pos[R_KNEE] = 2024
        side_split_pos[R_FOOT] = 3096
        side_split_pos[L_HIP_X] = 2072
        side_split_pos[L_HIP_Z] = 1057
        side_split_pos[L_HIP_Y] = 3072
        side_split_pos[L_KNEE] = 2033
        side_split_pos[L_FOOT] = 1035

        side_split_pos = side_split_pos[1:]

        durations = [5000]*24

        self.dynamixel_handler.move_many_servos(self.all_ids, side_split_pos, durations)

        time.sleep(6)

    def split(self):
        split_pos = [2048]*25
        split_pos[NECK_Y] = 1900
        split_pos[R_SHOULDER_Y] = 1469
        split_pos[R_SHOULDER_X] = 2975
        split_pos[R_SHOULDER_Z] = 2049
        split_pos[R_ELBOW] = 2320
        split_pos[L_SHOULDER_Y] = 1277
        split_pos[L_SHOULDER_X] = 1076
        split_pos[L_SHOULDER_Z] = 2131
        split_pos[L_ELBOW] = 1694
        split_pos[TORSO_Z] = 2023
        split_pos[TORSO_Y] = 2247
        split_pos[TORSO_X] = 2074
        split_pos[R_HIP_X] = 2050
        split_pos[R_HIP_Z] = 2013
        split_pos[R_HIP_Y] = 3015
        split_pos[R_KNEE] = 2026
        split_pos[R_FOOT] = 2931
        split_pos[L_HIP_X] = 2048
        split_pos[L_HIP_Z] = 2097
        split_pos[L_HIP_Y] = 3150
        split_pos[L_KNEE] = 2025
        split_pos[L_FOOT] = 1020

        split_pos = split_pos[1:]

        durations = [5000]*24

        self.dynamixel_handler.move_many_servos(self.all_ids, split_pos, durations)

        time.sleep(6)
    

    def pushup(self):
        pushup_pos = [2048]*25

        pushup_pos[L_ELBOW] = 1024
        pushup_pos[R_ELBOW] = 3072

        pushup_pos[R_SHOULDER_Y] = 3072
        pushup_pos[L_SHOULDER_Y] = 1024

        pushup_pos[R_SHOULDER_Z] = 1024
        pushup_pos[L_SHOULDER_Z] = 3072
        pushup_pos[NECK_Y] = 1325

        up_pos = pushup_pos.copy()
        up_pos[R_SHOULDER_X] = 2350
        up_pos[R_ELBOW] = 2700

        up_pos[L_SHOULDER_X] = 1750
        up_pos[L_ELBOW] = 1422

        pushup_pos = pushup_pos[1:]
        up_pos = up_pos[1:]

        durations = [5000]*24

        self.dynamixel_handler.move_many_servos(self.all_ids, pushup_pos, durations)

        input("Start push ups")
        print("Starting push ups...")
        
        for _ in range(2):
            self.dynamixel_handler.move_many_servos(self.all_ids, up_pos, durations)
            time.sleep(6)

            self.dynamixel_handler.move_many_servos(self.all_ids, pushup_pos, durations)
            time.sleep(6)
    
    def squat(self):
        squat_pos = [2048]*25

        squat_pos[NECK_Y] = 1800
        squat_pos[R_SHOULDER_Y] = 2872
        squat_pos[R_SHOULDER_X] = 2850
        squat_pos[R_SHOULDER_Z] = 2048
        squat_pos[R_ELBOW] = 2400
        squat_pos[L_SHOULDER_Y] = 1224
        squat_pos[L_SHOULDER_X] = 1250
        squat_pos[L_SHOULDER_Z] = 2064
        squat_pos[L_ELBOW] = 1850
        squat_pos[TORSO_Z] = 2048
        squat_pos[TORSO_Y] = 2400
        squat_pos[TORSO_X] = 2048
        squat_pos[R_HIP_X] = 2095 
        squat_pos[R_HIP_Z] = 2248
        squat_pos[R_HIP_Y] = 1448
        squat_pos[R_KNEE] = 3078
        squat_pos[R_FOOT] = 1751
        squat_pos[L_HIP_X] = 1991
        squat_pos[L_HIP_Z] = 1848
        squat_pos[L_HIP_Y] = 2648
        squat_pos[L_KNEE] = 1018
        squat_pos[L_FOOT] = 2350

        lean_backwards = squat_pos.copy()
        lean_backwards[TORSO_Y] = 2350

        base_positions = [2048]*25

        # Go to base position with arm down
        base_positions[R_SHOULDER_X] = 2848
        base_positions[L_SHOULDER_X] = 1248
        base_positions[R_ELBOW] = 2400
        base_positions[L_ELBOW] = 1850

        lean_forward = base_positions.copy()
        lean_forward[TORSO_Y] = 2100

        durations = [5000]*24
        lean_forward_durations = [3000]*24
        lean_backwards_durations = [7000]*24

        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions[1:], durations)

        time.sleep(6)

        input("Start")

        self.dynamixel_handler.move_many_servos(self.all_ids, lean_forward[1:], lean_forward_durations)
        
        squat_durations = [10000]*24

        self.dynamixel_handler.move_many_servos(self.all_ids, squat_pos[1:], squat_durations)

        time.sleep(11)

        input("Go up")

        self.dynamixel_handler.move_many_servos(self.all_ids, lean_backwards[1:], lean_backwards_durations)

        time.sleep(7)

        base_positions[TORSO_Y] = 2048
        base_positions[R_SHOULDER_Y] = 1900
        base_positions[L_SHOULDER_Y] = 2200

        first_durations = [8000]*24
        
        self.dynamixel_handler.move_many_servos(self.all_ids, base_positions[1:], first_durations)


        new_base_positions = base_positions.copy()
        new_base_positions[TORSO_Y] = 2040
        new_base_positions[R_SHOULDER_Y] = 2048
        new_base_positions[L_SHOULDER_Y] = 2048
        second_durations = [7000]*24
        time.sleep(4)

        self.dynamixel_handler.move_many_servos(self.all_ids, new_base_positions[1:], second_durations)
        
        time.sleep(7)