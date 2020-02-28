#coding=utf-8
from os import remove, rename, path
def change_coding(file, coding='GB18030',tmp_file_name='tmp'):
    """
    文件编码转换,将文件编码转换为UTF-8
    :param file:
    :return:
    """
    tmpfile = path.join(path.dirname(file), tmp_file_name)
    try:
        with open(file, 'r', encoding=coding) as fr, open(tmpfile, 'w', encoding='utf-8') as fw:
            content=fr.read()
            content=str(content.encode('utf-8'),encoding='utf-8')
            print(content,file=fw,end='')
    except UnicodeDecodeError as e:
        print(file+': ' + e.reason)
    remove(file)
    rename(tmpfile, file)

change_coding('./0222_ready/part-00000-test')