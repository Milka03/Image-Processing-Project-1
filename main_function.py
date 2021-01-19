import numpy as np
import cv2
import glob
import os
from hand import detect_hand


def save(image, name, path):
    name = os.path.basename(name)
    if path.find('boxes') != -1:
        name = 'BOX-' + name
    cv2.imwrite(path + '/' + name, image)

def make_masks():
    readPathNails = './images'
    writePathMask = './Results/masks'
    writePathBox = './Results/boxes'

    files = [f for f in glob.glob(readPathNails + "**/*.jpg", recursive=True)] 
    for f in files:
        mask, box = detect_hand(f, False)
        save(mask, f, writePathMask)
        save(box, f, writePathBox)
        print(f + " saved!")
    
    print("All files saved!\n")
    cv2.waitKey()


def compare_masks():
    make_masks()
    print("------------------------------------Comparison of masks------------------------------------\n")
    readPathLabel = './labels'
    myPathMasks = './Results/masks'  

    files_mask = [f for f in glob.glob(myPathMasks + "**/*.jpg", recursive=True)] 
    mean_IoU = 0
    mean_Dice = 0
    min_IoU = 1
    max_IoU = 0
    min_Dice = 1
    max_Dice = 0
    results = []
    photosCount = 0
    
    for f in files_mask:
        name = os.path.basename(f)
        mask = cv2.imread(myPathMasks + '/' + name)
        label = cv2.imread(readPathLabel + '/' + name)
        photosCount += 1

        Dice = np.sum(mask[label==255])*2.0 / (np.sum(mask) + np.sum(label))
        IoU = np.sum(cv2.bitwise_and(mask, label)) / np.sum(cv2.bitwise_or(mask, label))

        results.append((name, IoU, Dice))
        print(name + ' ->  IoU: ' + str(IoU) + ',   Dice: ' + str(Dice))
        mean_IoU += IoU
        mean_Dice += Dice
        if IoU > max_IoU:
            max_IoU = IoU
            nameMaxIoU = os.path.basename(f)
        if IoU < min_IoU:
            min_IoU = IoU
            nameMinIoU = os.path.basename(f)
        if Dice > max_Dice:
            max_Dice = Dice
            nameMaxDice = os.path.basename(f)
        if Dice < min_Dice:
            min_Dice = Dice
            nameMinDice = os.path.basename(f)
    
    mean_IoU /= photosCount
    mean_Dice /= photosCount
    print("\n---------------------------------------- Jaccard index ---------------------------------------\n")
    print("Mean IoU: " + str(mean_IoU))
    print("Min IoU: " + nameMinIoU + ' - ' + str(min_IoU))
    print("Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU))
    print("\n--------------------------------------- Dice coefficient -------------------------------------\n")
    print("Mean Dice: " + str(mean_Dice))
    print("Min Dice: " + nameMinDice + ' - ' + str(min_Dice))
    print("Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")
    with open('results_masks.txt', 'w') as f:
        f.write("--------------------------------- Comparison of masks ------------------------------\n\n")
        f.write("	            Filename: 	                  IoU score:          Dice Score:\n")
        for item in results:          
            f.write("%s\n" % str(item))
        f.write("\n%s\n\n" % "------------------------------------- Jaccard index --------------------------------")
        f.write("%s" % "Mean IoU: " + str(mean_IoU) + "\n")
        f.write("%s" % "Min IoU: " + nameMinIoU + ' - ' + str(min_IoU) + "\n")
        f.write("%s" % "Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU) + "\n")
        f.write("\n%s\n\n" % "------------------------------------ Dice coefficient ------------------------------")
        f.write("%s" % "Mean Dice: " + str(mean_Dice) + "\n")
        f.write("%s" % "Min Dice: " + nameMinDice + ' - ' + str(min_Dice) + "\n")
        f.write("%s" % "Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")

    cv2.waitKey()



if __name__ == '__main__':
    compare_masks()
