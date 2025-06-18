import yadisk
import pandas as pd

yadisk_token = 'token'
fileName = 'users_action'
local_path = 'tg_bot_project/'
disk_path = '/disk/'

def transfer_to_yadisk(local_file_path):
    y = yadisk.YaDisk(token = yadisk_token)
    df = pd.read_csv(f'{local_file_path}{fileName}.csv')
    df.to_excel(f'{local_file_path}{fileName}.xlsx')

    if y.check_token() :
        print('yadisk work')
    else :
        print('TOKEN NOT WORK')
    try :
        y.upload(f'{local_file_path}{fileName}.xlsx', f'{fileName}.xlsx', overwrite=True)
        print(f'File {local_file_path} upload on yadisk')
    except Exception as e :
        print(f'not upload file: {e}')

transfer_to_yadisk(local_path)          

