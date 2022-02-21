import pandas as pd

col_name_list = ['Arbitration_ID', 'Class']

# # O_Training
df = pd.read_csv('Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_2.csv', usecols=col_name_list)
# # 1_Submission
# df = pd.read_csv('Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/1_Submission/Pre_submit_D.csv.csv', usecols=col_name_list)
# 1_Final
# df = pd.read_csv('Car_Hacking_Challenge_Dataset_rev20Mar2021/1_Final/Fin_host_session_submit_S.csv', usecols=col_name_list)


count_Normal = 0
count_Flooding = 0
count_Spoofing = 0
count_Replay = 0
count_Fuzzing = 0
can_type = []
for i in range(len(df)):
    # if df.get('Class').values[i] != 'Normal':
    #     print(f'第 {i} 条报文数据: {df.iloc[i]}')
    #     break
    if df.get('Class').values[i] not in can_type:
        can_type.append(df.get('Class').values[i])

print(can_type)