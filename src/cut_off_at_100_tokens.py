
import os
import glob
from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/home/logan/shared/kaiqiang_to_turk', 'Path to folders containing summaries')

def main(unused_argv):

    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)
    # original_dir = dec_dir.split('/')
    # original_dir[-1] = 'reference'
    # rouge_ref_dir = '/'.join(original_dir)

    folders = os.listdir(FLAGS.data_dir)
    for folder in folders:
        files = glob.glob(os.path.join(FLAGS.data_dir, folder, '*'))
        for file in files:
            with open(file) as f:
                lines = f.readlines()
            count = 0
            out_str = ''
            for line in lines:
                should_break = False
                tokens = line.split()
                for t in tokens:
                    if count >= 100:
                        should_break = True
                        break
                    out_str += t + ' '
                    count += 1
                out_str += '\n'
                if should_break:
                    break
            with open(file, 'wb') as f:
                f.write(out_str)


if __name__ == '__main__':
    app.run(main)
















