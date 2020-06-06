import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
host_add = "***********.com"
host_password = "*************"
guest_add = "***************@gmail.com"
subject = "Achievement of Best Accuracy of model"
content = '''Hello Developer, 
		Congratulations your model has achieved best accuracy i.e accuracy is greater than 90%.
			Thank you ...'''
message = MIMEMultipart()
message['From'] = host_add
message['To'] = guest_add
message['Subject'] = subject
message.attach(MIMEText(content, 'plain'))
a_file=open('acc_file.txt','r')
data=a_file.read()
data=float(data)
a_file.close()
if data < 0.90:
	cnn_file=open('/dataset/file.py','r')
	all_lines=cnn_file.readlines()
	all_lines[7]=all_lines[7]+'\n'+all_lines[6]+'\n'+all_lines[7]+'\n'

	cnn_file=open('/dataset/file.py','w')
	cnn_file.writelines(all_lines)
	cnn_file.close()
else:
	
	session = smtplib.SMTP('smtp.gmail.com', 587)
	session.starttls()
	session.login(host_add, host_password)
	text = message.as_string()
	session.sendmail(host_add, guest_add , text)
	session.quit()
	print('Successfully sent your mail')
