import os
import cv2
import csv
import json
import argparse
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms

from detector import FaceBoxLandmarkDetector
from aligner import Aligner
from light_cnn_v4 import LightCNN_V4
from light_cnn import LightCNN_9Layers


def to_numpy(tensor):
    """
    Returns numpy array from torch tensor.
    :param tensor: Torch tensor
    :return: Returns numpy array from torch tensor.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_batch(batch: list):
        """
        Checking and preparing batch.

        The image is resized if it is larger than the fixed size,
        if it is smaller, it remains in the original resolution.

        :param batch: list of images.
        :return: original_batch: list of numpy.ndarray images with origin size
                 checked_batch: list of numpy.ndarray images with the changed size of the sides
                 ids_map: dict of mapping of valid and none valid images
                 scales_x: list of the scale factors of the width of the images
                 scales_y: list of the scale factors of the height of the images
        """
        checked_batch = []
        original_batch = []
        ids_map = {}
        new_id = 0
        scales_x = []
        scales_y = []
        for i, image in enumerate(batch):
            if image.shape[2] > 3:
                image = image[:, :, 0:3]

            # the calculation of scale factors
            scale_y = 1024 / image.shape[0]
            scale_x = 768 / image.shape[1]

            # the check increases the area when the size changes or decreases
            # if it decreases, we change the size; if it doesn't, we don't do anything
            if scale_y * scale_x < 1:
                image_resize = cv2.resize(image, None, fx=scale_x, fy=scale_y)
                checked_batch.append(image_resize)
            else:
                scale_x = 1.0
                scale_y = 1.0
                checked_batch.append(image)

            scales_x.append(scale_x)
            scales_y.append(scale_y)
            original_batch.append(image)
            ids_map[i] = new_id
            new_id += 1

        return original_batch, checked_batch, ids_map, scales_x, scales_y


def landmarks_scale(scales_x: float,
                    scales_y: float,
                    landmarks: list) -> list:
        """
        Scale the landmarks to original image size

        :param scales_x: scale factors for x-axis
        :param scales_y: scale factors for y-axis
        :param landmarks: landmarks of detected faces
        """

        scale_landmarks = []
        for s_x, s_y, lands in zip(scales_x, scales_y, landmarks):
            if s_x == s_y and s_x == 1.0:
                scale_landmarks.append(lands)
            else:
                for i, _ in enumerate(lands):
                    lands[i, :, 0] /= s_x
                    lands[i, :, 1] /= s_y

                scale_landmarks.append(lands)

        return scale_landmarks


def extract_face_embeddings(lightcnnVersion, image, face_detector, aligner):
    
    origin_size_batch, cv2_batch, ids_map, scales_x, scales_y = check_batch([image])
    bboxes, landmarks = face_detector.detect_faces(cv2_batch)
    #print(bboxes)
    #print(len(bboxes))
    #print(landmarks)

    if not len(bboxes[0]):
        return False, [], 'Retina failed'

    landmarks = landmarks_scale(scales_x, scales_y, landmarks)

    """
    for box, lnd in zip(bboxes[0], landmarks[0]):
        sr = (int(box[0]), int(box[1]))
        er = (int(box[2]), int(box[3]))
        cv2.rectangle(image, sr, er, (0, 0, 255), 2)

        for p in lnd:
            center = (int(p[0]), int(p[1]))
            cv2.circle(image, center, 1, (0, 0, 255), 2)

    
    cv2.imshow('image', image)
    k = cv2.waitKey(0)
    if k == 27:  # ESC to exit
        exit()
    cv2.destroyAllWindows()
    """

    alignedImage = aligner.align_face(image, landmarks[0][0])

    """
    cv2.imshow('alignedImage', alignedImage)
    k = cv2.waitKey(0)
    if k == 27:  # ESC to exit
        exit()
    cv2.destroyAllWindows()
    """

    # Create LightCNN Pytorch model

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0],),
    ])

    if lightcnnVersion == 'v4':
        lightcnnV4_model = LightCNN_V4()
        lightcnnV4_model.eval()
        checkpoint = torch.load('weights/lightcnn/LightCNN-V4_checkpoint.pth.tar')
        lightcnnV4_model.load_state_dict(checkpoint['state_dict'])
        lightcnnV4_model = lightcnnV4_model.cuda()

        alignedImage = transform(alignedImage)

        input = torch.zeros(1, 3, 128, 128)
        input[0,:,:,:] = alignedImage
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)

        with torch.no_grad():
            features = lightcnnV4_model(input_var)
            if not len(features):
                return False, [], 'LightCNN failed'

    elif lightcnnVersion == 'v1':


        model = LightCNN_9Layers(num_classes=79077)
        model.eval()
        model = torch.nn.DataParallel(model).cuda()

        checkpoint = torch.load('weights/lightcnn/lightcnn9.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

        alignedImage = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2GRAY)

        input = torch.zeros(1, 1, 128, 128)
        alignedImage   = transform(alignedImage)
        input[0,:,:,:] = alignedImage
        input = input.cuda()

        input_var   = torch.autograd.Variable(input, volatile=True)
        with torch.no_grad():
            _, features = model(input_var)
            if not len(features):
                return False, [], 'LightCNN failed'


    """
    print(features)
    print(len(features))
    print(len(features[0]))
    """

    return True, [to_numpy(features[0])], ''


def cosine_distance(a, b):
    a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def calculate_cosine_distances(lightcnnVersion, dataset_path, pairs_format_type, pairs_file_path, results_dir, image_color):

    """
    Calculats cosine distances between defined positive and negative pairs of images. 
    Calculated distances are written to distances.csv file

    Parameters:
        dataset_path          -  path image dataset
        pairs_file_path       -  path to file in which are defined pairs of people 
        results_dir           -  path of directory in which resulted distances wll be saved
    """

    
    face_detector = FaceBoxLandmarkDetector(
        model_weights = 'weights/detector/mnet.25',
        model_epoch = 0,
        gpu_device_list = 0,
    )

    aligner = Aligner(face_size=[128,128])
    

    pairs = [] # list of positive and negative pairs of images to be compaired. One pair: [person1-ImgN, person2-ImgN]
    with open(pairs_file_path, 'r') as f:
        data = f.read().split('\n')[:-1]
        data = [record.replace('\t', ' ') for record in data]
    print(data)

    for record in data:
        fields = record.split(' ')
        if len(fields) == 3:
            pairs.append([fields[0]+'&'+fields[1], fields[0]+'&'+fields[2]])
        if len(fields) == 4:
            pairs.append([fields[0]+'&'+fields[1], fields[2]+'&'+fields[3]])

    print(pairs)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
            
    distances_file_path = results_dir + os.path.sep + 'distances.csv'
    with open(distances_file_path, 'w', encoding='UTF8', newline='') as f:
        header = ['person1&ImgN', 'person2&ImgN', 'distance', 'Retina on person1', 'Retina on person2', 'LightCNN on person1', 'LightCNN on person2']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
    print('Writing results to ', distances_file_path)


    for pair in pairs:

        print(pair[0])
        print(pair[1])

        msg1_retina = ''
        msg1_lightcnn = ''
        msg2_retina = ''
        msg2_lightcnn = ''

        person1_imgNum = pair[0].split('&')
        person1_images = np.sort(glob(dataset_path + os.path.sep + person1_imgNum[0] + '/*'))
        #print(person1_images)
        img_path_person1 = person1_images[int(person1_imgNum[1]) - 1]
        print(img_path_person1)
        img_person1 = cv2.imread(img_path_person1)
        if image_color == 'bgr':
            img_person1 = cv2.cvtColor(img_person1, cv2.COLOR_RGB2BGR)
        #cv2.imshow('img_person1', img_person1)
        #cv2.waitKey(0)

        person2_imgNum = pair[1].split('&')
        person2_images = np.sort(glob(dataset_path + os.path.sep + person2_imgNum[0] + '/*'))
        #print(person2_images)
        img_path_person2 = person2_images[int(person2_imgNum[1]) - 1]
        print(img_path_person2)
        img_person2 = cv2.imread(img_path_person2)
        if image_color == 'bgr':
            img_person2 = cv2.cvtColor(img_person2, cv2.COLOR_RGB2BGR)
        #cv2.imshow('img_person2', img_person2)
        #cv2.waitKey(0)

        ret1, embeddings1, msg1 = extract_face_embeddings(lightcnnVersion, img_person1, face_detector, aligner)
        ret2, embeddings2, msg2 = extract_face_embeddings(lightcnnVersion, img_person2, face_detector, aligner)

        if ret1:
            msg1_retina = 'Successful'
            msg1_lightcnn = 'Successful'
            #print('embeddings1', embeddings1)
        else: 
            if msg1 == 'Retina failed': msg1_retina = 'Failed'
            elif msg1 == 'LightCNN failed': msg1_lightcnn = 'Failed'


        if ret2:
            msg2_retina = 'Successful'
            msg2_lightcnn = 'Successful'
            #print('embeddings2', embeddings2)
        else: 
            if msg2 == 'Retina failed': msg2_retina = 'Failed'
            elif msg2 == 'LightCNN failed': msg2_lightcnn = 'Failed'


        distance = -1
        if ret1 and ret2:
            distance = cosine_distance(embeddings1, embeddings2)[0]

        with open(distances_file_path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            data = [pair[0], pair[1], distance, msg1_retina, msg2_retina, msg1_lightcnn, msg2_lightcnn]
            writer.writerow(data)

        print('Distances are written to ', distances_file_path)



def calculate_metrics(distances_file_path, distance_threshold):
    print('Calculating metrics...')
    #print(distances_file_path)

    TP, FP, TN, FN = 0, 0, 0, 0
    data = []
    n_failed_comparisons = 0

    with open(distances_file_path, 'r') as f:
        #header = ['person1 (on DL card)', 'person2 (on camera images)', 'distances']
        reader = csv.DictReader(f, fieldnames=None)  # fieldnames=None to skip the header
        for row in reader:
            #print(row)

            if not row['distance'] == '[]':

                distance = float(row['distance'][1:-1])
                #print("distance", distance)
                 
                result = distance < distance_threshold
                #print("result", result)
               
                person1 = row['person1&ImgN'].split('&')[0]
                person2 = row['person2&ImgN'].split('&')[0]
                #print(person1, ' ', person2)
                #print("-------------")

                data.append([person1, person2, person1 == person2, bool(result)])

                if person1 == person2:       # positive pair (same person)
                    if result:
                        TP += 1
                    else:
                        FN += 1
                elif person1 != person2:     # negative pair (different people)
                    if result:
                        FP += 1
                    else:
                        TN += 1

            else:
                n_failed_comparisons += 1
            

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precision, recall = None, None
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        if (TP + FN) > 0:
            recall = TP / (TP + FN)

        FMR, FNMR = None, None
        if (FP + TN) > 0:
            FMR = FP / (FP + TN)
        if (TP + FN) > 0:
            FNMR = FN / (TP + FN) 

        """
        print('FP ', FP)
        print('FN ', FN)

        print('FMR ', FMR)
        print('FNMR ', FNMR)
        """

        TAR, TRR = None, None
        if (TP + FN) > 0:
            TAR = TP / (TP + FN)
        if (FP + TN) > 0:
            TRR = TN / (FP + TN)

        # Another way:
        #TAR = 1 - FNMR
        #TRR = 1 - FMR

        EER = None
        accuracy_based_on_eer = None

        if round(FMR,2) == round(FNMR,2) and FMR != 0.0:
            EER = FMR
            #print('EER found: ', EER)
            accuracy_based_on_eer = 1.0 - EER

        #return accuracy, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons
        return TP, FP, TN, FN, accuracy, precision, recall, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons



def results_for_fixed_threshold(distances_file_path, distance_threshold, results_dir):
    """
    Calculates metrics based on previously calculated distances between pairs and given distance threshold.
    """

    TP, FP, TN, FN, accuracy, precision, recall, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons = calculate_metrics(distances_file_path, distance_threshold)
    
    #save_dir = 'face_verification_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_name = results_dir + os.path.sep + 'results_distThr=' + str(distance_threshold)

    with open(results_file_name + '.txt', 'w') as f:
        f.write('Cosine distance threshold: ' + str(distance_threshold) + '\n')
        f.write('Accuracy: ' + str(accuracy) + '\n')
        
        f.write('TP: ' + str(TP) + '\n')
        f.write('FP: ' + str(FP) + '\n')
        f.write('TN: ' + str(TN) + '\n')
        f.write('FN: ' + str(FN) + '\n')

        f.write('Precision: ' + str(precision) + '\n')
        f.write('Recall: ' + str(recall) + '\n')

        f.write('FMR: ' + str(FMR) + '\n')
        f.write('FNMR: ' + str(FNMR) + '\n')
        f.write('TAR: ' + str(TAR) + '\n')
        f.write('TRR: ' + str(TRR) + '\n')

        if round(FMR,2) == round(FNMR,2):
            f.write('EER: ' + str(EER) + '\n')
            f.write('Accuracy based on EER (acc = 1 - EER) : ' + str(accuracy_based_on_eer) + '\n')

        f.write('Number of comparisons that failed (because face detection or embedding extraction failed): ' + str(n_failed_comparisons))

    with open(results_file_name+'.csv', 'w', encoding='UTF8', newline='') as f:
        header = ['person1 (on DL card)', 'person2 (on camera images)', 'person1 == person2 [actual]', 'person1 == person2 [predicted]']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    with open(results_file_name+'.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)



def find_error_rates_per_threshold(distances_file_path, results_dir):
    """
    Calculates error rates (FMR and FNMR) for multiple threshold values (which are in interval [0,1] with step 0.05).
    Resulted FMRs and FNMRs are written to json file and can be later used for finding EER.
    """

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    error_rates = []

    thresholds = np.arange(0, 1, 0.05)
    #print(thresholds)

    for thr in thresholds:
        print('Distance threshold: ', thr)
        TP, FP, TN, FN, accuracy, precision, recall, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons = calculate_metrics(distances_file_path, thr)

        error_rates.append({'distance_threshold': thr, 'accuracy':accuracy, 'FMR': FMR, 'FNMR':FNMR})

        if EER is not None:
            print('Found EER.')
            print(EER, accuracy_based_on_eer)

            with open(results_dir + os.path.sep + 'EER_info.txt', 'w') as f:
                f.write('Cosine distance threshold: ' + str(thr) + '\n')
                f.write('EER: ' + str(EER) + '\n')
                f.write('Accuracy based on EER (acc = 1 - EER) : ' + str(accuracy_based_on_eer) + '\n')
                
                f.write('Accuracy (calculated as (TP + TN) / (TP + TN + FP + FN)): ' + str(accuracy) + '\n')
                
                f.write('TP: ' + str(TP) + '\n')
                f.write('FP: ' + str(FP) + '\n')
                f.write('TN: ' + str(TN) + '\n')
                f.write('FN: ' + str(FN) + '\n')
                
                f.write('Precision: ' + str(precision) + '\n')
                f.write('Recall: ' + str(recall) + '\n')

                f.write('FMR: ' + str(FMR) + '\n')
                f.write('FNMR: ' + str(FNMR) + '\n')
                f.write('TAR: ' + str(TAR) + '\n')
                f.write('TRR: ' + str(TRR) + '\n')
                
    #print(error_rates)
    with open(results_dir + os.path.sep + 'error_rates.json', 'w') as outfile:
        json.dump(error_rates, outfile)



def find_EER(error_rates_file_path, results_dir):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_name = results_dir + os.path.sep + 'EER_threshold.txt'

    thresholds, FMRs, FNMRs = [], [], []

    with open(error_rates_file_path) as json_file:
        data = json.load(json_file)
        for field in data:
            thresholds.append(field['distance_threshold'])
            FMRs.append(field['FMR'])
            FNMRs.append(field['FNMR'])

    thresholds, FMRs, FNMRs = np.array(thresholds), np.array(FMRs), np.array(FNMRs) 

    #idx = np.argwhere(np.diff(np.sign(FMRs - FNMRs))).flatten()
    #idx = np.argwhere(np.sign(FMRs - FNMRs) == 0).flatten()
    idx = np.argwhere(np.round(np.abs(FMRs - FNMRs), 1) == 0).flatten()
    print('EER idx', idx)

    plt.figure(figsize=(120,110))
    plt.plot(thresholds, FMRs, label='FMR')
    plt.plot(thresholds, FNMRs, label='FNMR')
    plt.xticks(thresholds)
    plt.yticks(FMRs)
    plt.yticks(FNMRs)
    plt.xlabel('thresholds')
    plt.ylabel('error rates')
    plt.legend(["FMR", "FNMR"], loc ="best")

    
    plt.plot(thresholds[idx], FMRs[idx], 'ro')

    plt.savefig(results_dir + os.path.sep + 'error_rates.png')
    #plt.show()

    EER = FMRs[idx[0]]
    thr = thresholds[idx[0]]
    acc = 1.0 - EER

    with open(results_file_name, 'w') as f:
        f.write('EER: ' + str(EER) + '\n')
        f.write('Accuracy based on EER (acc = 1-EER) : ' + str(acc) + '\n')
        f.write('threshold: ' + str(thr))



def plot_accuracy(error_rates_file_path, results_dir):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_name = results_dir + os.path.sep + 'max_accuracy_threshold.txt'

    thresholds, accuracies = [], []

    with open(error_rates_file_path) as json_file:
        data = json.load(json_file)
        for field in data:
            thresholds.append(field['distance_threshold'])
            accuracies.append(field['accuracy'])

    thresholds, accuracies = np.array(thresholds), np.array(accuracies)

    maxAccIdx = np.argmax(accuracies)
    print('maxAccIdx', maxAccIdx)
    maxAcc = accuracies[maxAccIdx]
    maxAccThreshold = thresholds[maxAccIdx]

    plt.figure(figsize=(120,110))
    plt.plot(thresholds, accuracies, label='FMR')
    plt.xticks(thresholds)
    plt.yticks(accuracies)
    plt.xlabel('thresholds')
    plt.ylabel('error rates')
    plt.legend(["accuracy"], loc ="best")
    
    plt.plot(maxAccThreshold, maxAcc, 'ro')

    plt.savefig(results_dir + os.path.sep + 'accuracy.png')

    with open(results_file_name, 'w') as f:
        f.write('Max accuracy [calculated as (TP + TN) / (TP + TN + FP + FN)] : ' + str(maxAcc) + '\n')
        f.write('Threshold that gives max accuracy : ' + str(maxAccThreshold))

def main():

    parser = argparse.ArgumentParser(description='face verification')

    parser.add_argument('--lightcnnVersion', type=str, default='v4', required=False, help='version of lightcnn')

    parser.add_argument('--calculate_distances', default=False, action='store_true', help='flag to indicate calculation of distances between samples')
    parser.add_argument('--dataset_path', type=str, required=False, help='path to image dataset')
    parser.add_argument('--pairs_format_type', type=str, default='lfw', required=False, help='Format in which pairs file is organized. Options: lfw, calfw')
    parser.add_argument('--pairs_file_path', type=str, required=False, help='File with positive and negative pairs of people for testing')
    parser.add_argument('--image_color', type=str, default='bgr', required=False, help='Color in which images will be converted. Possible values: rgb, bgr')
    
    parser.add_argument('--find_error_rates', default=False, action='store_true', help='flag to indicate calculating error rates per thresholds in [0,1]')
    parser.add_argument('--distances_file_path', type=str, required=False, help='path to file that contains distances')

    parser.add_argument('--fixed_distance_threshold', default=False, action='store_true', help='flag to indicate calculation of metrics based on fixed distance threshold')
    parser.add_argument('--distance_threshold', type=float, required=False, help='Distance threshold for ArcFace')

    parser.add_argument('--find_EER', default=False, action='store_true', help='flag to indicate searching for threshold that gives EER')
    parser.add_argument('--error_rates_file_path', type=str, required=False, help='path to json file that contains error rates')

    parser.add_argument('--plot_accuracy', default=False, action='store_true', help='flag to indicate plotting accuracy and searching for threshold that gives best accuracy')

    parser.add_argument('--results_dir', type=str, required=True, help='path to directory in which results will be (or are) saved')

    args = parser.parse_args()

    if args.calculate_distances:
        calculate_cosine_distances(args.lightcnnVersion, args.dataset_path, args.pairs_format_type, args.pairs_file_path, args.results_dir, args.image_color)

    if args.find_error_rates:
        find_error_rates_per_threshold(args.distances_file_path, args.results_dir)

    if args.fixed_distance_threshold:
        results_for_fixed_threshold(args.distances_file_path, args.distance_threshold, args.results_dir)

    if args.find_EER:
        find_EER(args.error_rates_file_path, args.results_dir)

    if args.plot_accuracy:
        plot_accuracy(args.error_rates_file_path, args.results_dir)

if __name__ == "__main__":
    main()