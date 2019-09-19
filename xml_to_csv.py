import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     round(float(member[4][0].text)),
                     round(float(member[4][1].text)),
                     round(float(member[4][2].text)),
                     round(float(member[4][3].text)),
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def convert(folder):
    ann_path = os.path.join(os.getcwd(), 'data', folder, 'annotations')

    if len(os.listdir(ann_path)) != 0:
        xml_df = xml_to_csv(ann_path)
        xml_df.to_csv('csv/' + folder + '_labels.csv', index=None)
        print(folder + ' folder successfully converted xml to csv!')
    else:
        print(folder + ' folder is empty!')


def main():
    # convert('train')
    # convert('validation')
    # convert('test')
    pass


main()
