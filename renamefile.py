import os

CURRENT_DIR = os.getcwd()

def main():
    DIR = os.path.join(CURRENT_DIR, 'images/test')
    for xml in os.listdir(DIR):
        file = xml.split('.')
        if len(file) == 2:
            name = str(file[0]) + '.' + str(file[1]) + '.xml'
            # print(name)

            # old = os.path.join(DIR, xml)
            # new = os.path.join(DIR, name)
            # os.rename(old, new)

if __name__ == '__main__':
    main()
