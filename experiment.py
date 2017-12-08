import multiprocessing
import sys, getopt
import os



def my_run(file_name, index, translate, dropout_prob, add_intrinsic_reward, srt_type):
    save_dir = '/home/ramin/CSC2541/ram_modified/resluts_newset/' + file_name.strip('.py')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + '/' + index + '_' + dropout_prob + add_intrinsic_reward + srt_type
    os.system('python {} --save_path {} --translate {} --dropout_prob {}'
              ' --add_intrinsic_reward {} --srt_type {}'.format(file_name, save_path,
                                                                translate, dropout_prob,
                                                               add_intrinsic_reward, srt_type
                                                                ))

if __name__ == '__main__':

    jobs = []
    for index in range(10):

        translate = str(0)
        dropout_prob = str(0.5)
        add_intrinsic_reward = 'False'
        srt_type = 'D'
        file_name = 'ram_srt.py'

        p = multiprocessing.Process(target=my_run, args = (file_name, str(index),
                                                           translate, dropout_prob,
                                                           add_intrinsic_reward, srt_type
                                                            ))

        jobs.append(p)
        p.start()

