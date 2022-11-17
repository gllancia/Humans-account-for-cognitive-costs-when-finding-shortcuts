import pandas as pd

def openExpFile(filename,skiprows,box_size=8.):
    with open(filename) as file:
        lines = file.readlines()
    
    df = []

    subject = lines[1].strip("ParticipantNo, ").strip("\n")

    lines = lines[skiprows:]
    for line in lines:
        if line[0] == "!":
            dsp = line.strip("!").strip("\n")
            trial,exp_type,dsp_goal  = dsp.split("_")
            exp_type = exp_type.strip("dsp")
        else:
            line = line.strip("\n")
            line = line.replace("box","")
            line = line.replace(" ","")

            data = line.split(",")

            if len(data) == 6:
                time,x,y,angle,box_x,box_y = data
                new_box_x = round(float(x)/box_size)
                new_box_y = round(float(y)/box_size)
                
                df.append([subject,int(trial),int(exp_type),int(dsp_goal),
                            float(time),float(x),float(y),float(angle),
                            int(new_box_x),int(new_box_y)])
                
    df = pd.DataFrame(df, columns=["subject","trial","exp_type","dsp_goal",
                                   "time","x","y","angle","box_x","box_y"])
    return df