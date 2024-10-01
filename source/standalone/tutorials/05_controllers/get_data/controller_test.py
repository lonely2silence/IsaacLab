from get_data import triad_openvr  #此处，如果在diff中运行，那么他的运行路径是在05——controllar底下，所以不能直接索引到triad_openvr
import time
import sys


def main():

    global diff
    diff = None

    v = triad_openvr.triad_openvr()
    v.print_discovered_objects()

    if len(sys.argv) == 1:
        interval = 1 #更新频率
    elif len(sys.argv) == 2:
        interval = 1/float(sys.argv[1])
    else:
        print("Invalid number of arguments")
        interval = False

    previous_quaternion = None    
    if interval:
        for _ in range(3):
            start = time.time()
            txt = ""
            current_quaternion = v.devices["controller_1"].get_pose_quaternion()
            if previous_quaternion is None:
                previous_quaternion = current_quaternion
            for each in current_quaternion:
                txt += "%.4f" % each
                txt += " "    
            
            diff_txt = "Diff: "
            
            
            diff = [a - b for a, b in zip(current_quaternion, previous_quaternion)]
           
            #print(diff)
            
            #print("\r" + txt + " | " + diff_txt, end="")

            previous_quaternion = current_quaternion

            time.sleep(0.5)

    return diff

         

