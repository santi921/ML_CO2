from pathlib import Path
import os


dir = "./xyz/DB3/"
#opens folders only
basepath = Path(dir)
sub_folders = []
for entry in basepath.iterdir():
    if entry.is_dir():
        sub_folders.append(entry.name)
print(sub_folders)

if(sub_folders != []):
    for i in sub_folders:
        dir_temp = dir + i + "/"
        print(dir_temp)
        dir_str = "ls " + str(dir_temp) +" | sort -d"
        temp = os.popen(dir_str).read()
        temp = str(temp).split()
        for names in temp:
            os.rename(dir+ i + "/" +names, dir+i+"_"+names)
