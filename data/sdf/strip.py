import os 
files = os.listdir()
for fil in files:
	with open(fil, "r+") as f:
		d = f.readlines()
		f.seek(0)
		try: 
			int(d[3].split()[0])
		except:
			print(fil)


		#for i in d:
		#	if i[0:4] != "Warn":
		#		f.write(i)
		#f.truncate()
