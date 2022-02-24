import os
import smtplib
import time
import mimetypes
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.header import Header


def send_email(username, password, sender, receivers, smtp_server, port, content, file_path):
    message = MIMEMultipart()

    title = os.path.basename(file_path)
    # 邮件主题
    subject = title
    message['Subject'] = subject
    # 邮件正文
    message.attach(MIMEText(content, 'plain', _charset='utf-8'))
    # 附件
    print(f'file_path {file_path}')
    if file_path:
        data = open(file_path, 'rb')
        ctype, encoding = mimetypes.guess_type(file_path)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        file_msg = MIMEBase(maintype, subtype)
        file_msg.set_payload(data.read())
        data.close()
        encoders.encode_base64(file_msg)  # 把附件编码
        file_msg.add_header('Content-Disposition', 'attachment', filename=title + ".zip")  # 修改邮件头
        message.attach(file_msg)

    try:
        server = smtplib.SMTP_SSL(smtp_server, port)
        server.login(username, password)
        server.sendmail(sender, receivers, message.as_string())
        server.quit()
        print("发送成功")
    except Exception as err:
        print("发送失败")
        print(err)

if __name__ == '__main__':
    smtp_server_ = "smtp.qq.com"
    username_ = "976362661@qq.com"
    password_ = "vsgbohogiqerbcji"
    sender_ = '976362661@qq.com'
    receivers_ = '976362661@qq.com'
    port_ = 465
    send_email(username_, password_, sender_, receivers_, smtp_server_, port_, '测试邮件', '/Users/aaron/Hebut/征稿_图像信息安全_20211130截稿/源程序/图塌缩分类/log/drive-download-20220221T024621Z-001.zip')