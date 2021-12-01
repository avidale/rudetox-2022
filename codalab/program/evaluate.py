import os.path
import sys
import requests


def get_submitted(parent):
    names = [name for name in os.listdir(parent)]
    if len(names) == 0:
        raise RuntimeError('No files in submitted')
    if len(names) > 1:
        if names[0] != "metadata" and names[1] != "metadata":
            raise RuntimeError('Multiple files in submitted: {}'.format(' '.join(names)))
    result = names[0] if names[0] != "metadata" else names[1]
    return os.path.join(parent, result)


def get_reference(parent):
    names = [os.path.join(parent, name) for name in os.listdir(parent)]
    if len(names) == 0:
        raise RuntimeError('No files in reference')
    if len(names) != 1:
        raise RuntimeError('There should be exact one file in reference: {}'.format(' '.join(names)))
    return names[0]


input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    truth_file = get_reference(truth_dir)
    truth = open(truth_file).readlines()

    submission_answer_file = get_submitted(submit_dir)
    transferred_sentences = open(submission_answer_file).readlines()

    result = requests.post(
            url='http://51.250.2.75:10301',
            json={"original_texts": truth, "rewritten_texts": transferred_sentences}, verify=False
        ).json()

    output_file.write('accuracy: ' + str(result['accuracy']) + '\n')
    output_file.write('similarity: ' + str(result['similarity']) + '\n')
    output_file.write('fluency: ' + str(result['fluency']) + '\n')
    output_file.write('joint: ' + str(result['joint']) + '\n')
    output_file.close()
