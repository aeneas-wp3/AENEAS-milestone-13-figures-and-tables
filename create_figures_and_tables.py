""" This Program calculates the number of data products produced by a
list of projects. The list of projects should be provided in csv format (See
examples/input.csv). """


import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bits_per_pixel = 32
bytes_per_pixel = bits_per_pixel*0.125
bits_per_vis = 64
bytes_per_vis = 8


def mulitplication_factor(nchan):
    return ((7*nchan)+9)


def parse_args():
    """ Parse input arguments."""
    parser = argparse.ArgumentParser(description='Calculate number of data \
                                     products')
    parser.add_argument('--i', dest='csv_file',
                        help='csv file containing required values for each \
                        project')
    args = parser.parse_args()

    return args


def load_file(file_name):
    df = pd.read_csv(file_name)
    df = df.fillna(0)
    return df


def storage_single_image(df):
    df['size single image'] = mulitplication_factor(df['number of channels']) \
                              * df['number of pixels']*df['number of pixels'] \
                              * df['number polarisations']*bytes_per_pixel/1e12

    return 0


def total_image_storage(df):
    df['total image storage (PB)'] = df['size single image'] \
                               * df['total time (hrs)'] \
                               / (df['length of single observation (hrs)']*1e3)
    df['total image storage (PB)'] = df['total image storage (PB)'].round(decimals=1)
    return 0


def size_single_visibility(df):
    df['size single visibility'] = df['number antennas']\
                                  * ((df['number antennas']-1)/2)\
                                  * (3600
                                     * df['length of single observation (hrs)']
                                     / df['time resolution (s)']) \
                                  * df['number polarisations']*bytes_per_vis \
                                  * df['number of channels vis']/1e12
    return 0


def total_vis_storage(df):
    df['total vis storage (PB)'] = df['size single visibility'] \
                                  * df['total time (hrs)'] \
                                  / (df['length of single observation (hrs)']
                                     * 1e3)
    df['total vis storage (PB)'] = df['total vis storage (PB)'].round(decimals=1)
    return 0


def nip_storage(df):
    df['NIP_storage (PB)'] = df['NIP data rate (Gbits/s)']*3600 \
                             * df['total time (hrs)']*1e9*0.125/1e15
    df['NIP_storage (PB)'] = df['NIP_storage (PB)'].round(decimals=1)
    return 0


def advanced_products_storage(df):
    df['Advanced data products (PB)'] = df['total image storage (PB)']*3 \
                                        + df['total vis storage (PB)']*3 \
                                        + df['NIP_storage (PB)']*3
    df['Advanced data products (PB)'] = df['Advanced data products (PB)'].round(decimals=1)
    df['Total storage (PB)'] = df['total image storage (PB)']*4 \
        + df['total vis storage (PB)']*4 \
        + df['NIP_storage (PB)']*4
    return 0


def number_of_years_running(df):
    df['number of years running'] = np.ceil(data_frame['total time (hrs)']
                                            / (data_frame['fraction \
observing time']*365*24)).astype('int')
    return 0


def calculate_data_per_year(df):
    df['data per year (PB)'] = df['Total storage (PB)']\
                          / df['number of years running']
    return 0


def calculate_num_image_data_products(df):
        factor = 8
        df['total_num_images'] = (df['total time (hrs)']
                                  / df['length of single observation (hrs)'])\
            * factor
        return 0


def calculate_num_image_data_products_per_year(df):
    df['num_images_per_year'] = df['total_num_images'] \
         / df['number of years running']
    return 0


def calculate_num_vis_data_products(df):
        df['total_num_vis'] = (df['total vis storage (PB)']
                               / df['size single visibility'])
        df['total_num_data_products'] = df['total_num_vis'] \
            + df['total_num_images']
        return 0


def calculate_num_vis_data_products_per_year(df):
    df['num_vis_per_year'] = df['total_num_vis'] \
         / df['number of years running']
    return 0


def table_1(df, file_name):
    num_rows = df.shape[0]
    vis_cols = ['name', 'channel res (kHz)', 'time resolution (s)',
                'number of channels vis', 'number of pointings',
                'number polarisations', 'total time (hrs)', 'number of pixels',
                'total vis storage (PB)']
    nip_cols = ['name', 'channel res (kHz)', 'time resolution (s)',
                'number of channels', 'number of pointings',
                'number polarisations', 'total time (hrs)', 'number of pixels',
                'NIP_storage (PB)']
    image_cols = ['name', 'channel res (kHz)',
                  'length of single observation (hrs)',
                  'number of channels',
                  'number of pointings', 'number polarisations',
                  'total time (hrs)', 'number of pixels',
                  'total image storage (PB)']
    just_vis = df.loc[df['total vis storage (PB)'] != '-']
    just_im = df.loc[df['total image storage (PB)'] != '-']
    just_nip = df.loc[df['NIP_storage (PB)'] != '-']
    vis_table = just_vis[vis_cols].copy()
    num_vis = vis_table.shape[0]
    im_table = just_im[image_cols].copy()
    num_im = im_table.shape[0]
    nip_table = just_nip[nip_cols].copy()
    num_nip = nip_table.shape[0]
    file = open(file_name, 'w')
    file.write("Data Product & Project & channel res. & time res. & \# channels &\
\# fields & \# pol. &  int. time & image size & \
                 SRC storage (PBytes)\\ \n")
    file.write('\\\\\\hline\n')
    file.write('\\\\\\hline\n')
    file.write("Vis. Products")
    for x in range(0, num_vis):
        file.write(" & %s \\\\\n" % " & ".join([str(val)
                                               for val in vis_table.iloc[x]]))
    file.write('\\\\\\hline\n')
    file.write("Image Products")
    for x in range(0, num_im):
        file.write(" & %s \\\\\n" % " & ".join([str(val)
                                               for val in im_table.iloc[x]]))
    file.write('\\\\\\hline\n')
    file.write("NIP")
    for x in range(0, num_nip):
        file.write(" & %s \\\\\n" % " & ".join([str(val)
                                               for val in nip_table.iloc[x]]))
    file.write('\\\\\\hline\n')
    file.write("Advanced Products")
    for x in range(0, num_rows):
        file.write(" & %s & - & - & - & - & - & - & - \
& %s \\\\\n" % (df['name'][x], df['Advanced data products (PB)'][x]))
    file.write('\\\\\\hline\n')
    file.write("Total Storage & - & - & - & - & - & - & - & - & %s \\\\\n" %
               df['Total storage (PB)'].sum())
    file.close()
    return 0


def calc_timeline(df, col, eor_col):
    df_temp = df.replace('-', 0)
    index = np.arange(16)
    non_eor_colnames = df_temp.loc[df_temp['is \
EoR'] == 0]['name'].values.tolist()
    eor_colnames = df_temp.loc[df_temp['is EoR'] == 1]['name'].values.tolist()
    columns = ['EoR'] + non_eor_colnames
    timeline_frame = pd.DataFrame(index=index, columns=columns)
    eor_data_per_year = df_temp[eor_col
                                ].loc[df_temp['name'].isin(eor_colnames)
                                      ].sum()/15
    timeline = np.zeros(16)
    timeline[1:] = eor_data_per_year
    timeline = timeline * np.arange(16)
    timeline_frame['EoR'] = timeline
    for x in index:
        name = df.at[x, 'name']
        timeline = np.zeros(16)
        if name not in eor_colnames:
            timeline[1:df.at[x, 'number of years running']+1] = \
                df.at[x, col]
            timeline = timeline * np.arange(16)
            timeline[df.at[x, 'number of years running']+1:] = max(timeline)
            timeline_frame[name] = timeline
    return timeline_frame


def plot_timeline(timeline, column, ax, c, style):
    ax.plot(timeline[column], color=c,
            linestyle=style, label=column)
    return


def make_figure(timeline, column_list, color_list, style_list, outname, ylabel):
    ax1 = plt.subplot(111)
    for x in range(0, len(column_list)):
        plot_timeline(timeline, column_list[x], ax1, color_list[x],
                      style_list[x])
    ax1.set_xlim(0, 15)
    ax1.set_xlabel('year')
    ax1.set_ylabel(ylabel)
    ax1.legend()
    plt.savefig(outname)
    plt.show()
    return 0


def table_2(timeline, file_name, unit=''):
    nrows = np.shape(timeline)[0]
    text = " "+unit+" & "
    file = open(file_name, 'w')
    file.write("year & %s %s &\\\\\n" %
               (text.join(timeline.columns.tolist()[0:7]),unit))
    for x in range(1, nrows):
        file.write("%s & %s &\\\\\n" % (str(x),
                   " & ".join(map(str, [round(float(i), 2)
                                        for i in timeline.iloc[x].values][0:7]))))
    file.write('\\hline\\\\\n')
    file.write("year & %s %s & Total %s\\\\\n" %
               (text.join(timeline.columns.tolist()[7:]), unit, unit))
    for x in range(1, nrows):
        file.write("%s & %s & % s \\\\\n" % (str(x),
                   " & ".join(map(str, [round(float(i), 2)
                                        for i in timeline.iloc[x].values][7:])),
                                        str(round(float(
                                         timeline.sum(axis=1)[x]), 2))))
    return 0


if __name__ == '__main__':
    args = parse_args()
    data_frame = load_file(args.csv_file)
    storage_single_image(data_frame)
    data_frame = data_frame.fillna(0)
    total_image_storage(data_frame)
    data_frame = data_frame.fillna(0)
    size_single_visibility(data_frame)
    data_frame = data_frame.fillna(0)
    total_vis_storage(data_frame)
    data_frame = data_frame.fillna(0)
    nip_storage(data_frame)
    data_frame = data_frame.fillna(0)
    advanced_products_storage(data_frame)
    number_of_years_running(data_frame)
    calculate_data_per_year(data_frame)
    calculate_num_image_data_products(data_frame)
    calculate_num_image_data_products_per_year(data_frame)
    calculate_num_vis_data_products(data_frame)
    calculate_num_vis_data_products_per_year(data_frame)
    data_frame = data_frame.replace(0, '-')
    timeline_data_storage = calc_timeline(data_frame, col='data per year (PB)',
                                          eor_col='Total storage (PB)')
    timeline_data_products = calc_timeline(data_frame,
                                           col='num_images_per_year',
                                           eor_col='total_num_data_products')
    timeline_data_products = timeline_data_products.drop(['HPSO4', 'HPSO5',
                                                          'HPSO18'], axis=1)
    timeline_data_products = timeline_data_products.round()
    table_1(data_frame, 'table_1.txt')
    table_2(timeline_data_storage, "table_2.txt", '(PB)')
    table_2(timeline_data_products, "table_3.txt", '')
    timeline_data_storage['Total'] = timeline_data_storage.sum(axis=1)
    timeline_data_products['Total'] = timeline_data_products.sum(axis=1)
    timeline_data_products = timeline_data_products.round()
    make_figure(timeline_data_storage,
                column_list=timeline_data_storage.columns.tolist()[1:4]
                + timeline_data_storage.columns.tolist()[6:11],
                color_list=['gold', 'gold', 'gold', 'blue', 'blue', 'blue',
                            'green', 'green'],
                style_list=['-', '--', '-.', '-', '--', '-.', '-', '--'],
                outname='timeline1.pdf', ylabel='storage requirments (PB)')
    make_figure(timeline_data_storage,
                column_list=timeline_data_storage.columns.tolist()[11:14],
                color_list=['green', 'green', 'orange'],
                style_list=['-', '--', '-'], outname='timeline3.pdf',
                ylabel='storage requirments (PB)')
    make_figure(timeline_data_storage,
                column_list=list(timeline_data_storage.columns.tolist()[i]
                                 for i in [0, 4, 5]),
                color_list=['black', 'red', 'purple'],
                style_list=['-', '-', '-'], outname='timeline2.pdf',
                ylabel='storage requirments (PB)')

    make_figure(timeline_data_storage,
                column_list=[timeline_data_storage.columns.tolist()[-1]],
                color_list=['black'],
                style_list=['-'],
                outname='timeline4.pdf', ylabel='storage requirments (PB)')

    make_figure(timeline_data_products,
                column_list=list(timeline_data_products.columns.tolist()[i]
                                 for i in [1, 2, 4, 6, 7, 9, 10]),
                color_list=['gold', 'gold', 'red', 'blue', 'blue', 'green',
                            'green'],
                style_list=['-', '--', '-', '-', '--', '-', '--'],
                outname='timeline_storage_1.pdf',
                ylabel='number of data products')

    make_figure(timeline_data_products,
                column_list=list(timeline_data_products.columns.tolist()[i]
                                 for i in [0, 3, 8]),
                color_list=['black', 'gold', 'blue'],
                style_list=['-', '-.', '-.'],
                outname='timeline_storage_2.pdf',
                ylabel='number of data products')

    make_figure(timeline_data_products,
                column_list=list(timeline_data_products.columns.tolist()[i]
                                 for i in [5]),
                color_list=['purple'],
                style_list=['-'],
                outname='timeline_storage_3.pdf',
                ylabel='number of data products')

    make_figure(timeline_data_products,
                column_list=list(timeline_data_products.columns.tolist()[i]
                                 for i in [-1]),
                color_list=['black'],
                style_list=['-'],
                outname='timeline_storage_4.pdf',
                ylabel='number of data products')
