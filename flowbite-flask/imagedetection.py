# run this file and look at output in terminal. We need to find a way
# to determine if the snake that is detected is venomous or not
# add add all these files to the website
from imageai.Classification.Custom import CustomImageClassification
import os
from snakedescription import snakeDescription
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd

# Read in information from snakes csv file to get poisonous data
dataSet = pd.read_csv('train.csv')


execution_path = os.getcwd()

# Using our own custom trained resnet-50 model
prediction = CustomImageClassification() # create an instance of the ImageClassification class from the ImageAI library
prediction.setModelTypeAsResNet50() # set the model type that the ImageClassification object will use to ResNet-50
#prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth")) # set the path to the model weights file that the ImageClassification object will use

# Set our model path
prediction.setModelPath(os.path.join(execution_path, "resnet50-train_local3-test_acc_0.42161_epoch-114.pt"))
prediction.setJsonPath(os.path.join(execution_path, "train_local3_model_classes.json"))

prediction.loadModel() # load the model with the specified model type (ResNet-50) and model weights (from the included .pth file)

snake_dict = {18: "Copperhead", 20: "Coppermouth", 25: "Green Vine Snake", 26: "Asian Vine Snake", 
              39: "Glossy Snake", 41: "Black-Headed Python", 48: "Thickhead Ground Snake",
              52: "Lowland Copperhead", 54: "Puff Adder", 57: "Gaboon Viper", 
              60: "Boa constrictor", 65: "Trans-Pecos Rat Snake", 71: "Brown tree snake", 
              73: "Kelung cat snake", 83: "Kelung Cat Snake", 87: "Fer-de-lance", 
              88: "Common Lancehead", 99: "Banded Krait", 110: "Eastern Worm Snake", 111: "Western Worm Snake", 
              113: "Rhombic Night Adder", 114: "Scarlet Snake", 122: "Rubber Boa", 135: "Golden Tree Snake", 
              140: "Kirtland's Snake", 155: "Sharp-Tailed Snake", 158: "Emerald Tree Boa", 
              159: "Amazon Tree Boa", 162: "Southern Smooth Snake", 163: "Eastern Diamondback Rattlesnake", 165: "Western Diamondback Rattlesnake", 
              168: "Sidewinder", 169: "Arizona Black Rattlesnake", 175: "Rock Rattlesnake", 177: "Black-Tailed Rattlesnake", 
              180: "Northern Black-Tailed Rattlesnake", 185: "Red Diamond Rattlesnake", 186: "Mojave Rattlesnake", 188: "Panamint Rattlesnake", 
              189: "Tiger Rattlesnake", 191: "Dusky rattlesnake", 193: "Prairie rattlesnake", 
              195: "Red-Lipped Snake", 203: "Russell's Viper", 215: "Painted Bronzeback", 216: "Green Tree Snake", 
              220: "Black Mamba", 226: "Southern Ring-necked Snake", 238: "Eastern Indigo Snake", 255: "Steppe Rat Snake", 
              263: "The Rainbow Boa", 280: "The Green Anaconda", 284: "Eastern Mudsnake", 302: "Red-Tailed Green Rat Snake", 
              315: "Horseshoe Whip Snake", 319: "Western Hognose Snake", 321: "Southern Hognose Snake", 323: "Green Whip Snake", 
              335: "Night Snake", 338: "Blunthead Tree Snake", 345: "Gray-Banded Kingsnake", 348: "Prairie Kingsnake", 
              352: "Eastern Kingsnake", 360: "Sonoran Mountain Kingsnake", 758 : "Checkered keelback", 755 : "Smooth earth snake", 751 : "Vipera seoanei",
              747 : "Vipera aspis", 746: "Horned Viper", 741: "Wagler'S pit viper", 740: "Bornean keeled green pit viper", 738: "lined snake", 725 : "bamboo pit viper", 701 : "Plains garter snake",
              699 : "Western ribbon snake",  698 : "Northwestern garter snake", 696 : "Checkered garter snake", 691 : "Black-necked gartersnake",
              690 : "Sierra garter snake", 686 : "Aquatic garter snake", 678 : "Western black-headed snake", 675 : "southwestern blackhead snake",
              674 : "Flat-headed snake", 672 : "Southeastern crowned snake", 656 : "Chicken snake",  652 : "pygmy rattlesnake", 651 : "eastern massasauga",
              634 : "Senticolis" , 629 : "western patch-nosed snake", 628 : "eastern patch-nosed snake", 623 : "Long-nosed snake", 619 : "Pine woods snake", 
              617 : "tiger keelback", 616 : "red-necked keelback", 609 : "Queen snake", 605 : "Ball python", 603 : "Indian python", 590 : "Eastern brown snake",
              578 : "Red-bellied black snake", 576 : "king brown snake", 575 : "Mole snake", 562 : "common mock viper", 560 : "brown-spotted pit viper",
              546 : "Mexican bullsnake", 545 : "Gopher snake", 544 : "spotted leafnose snake", 515 : "coastal taipan", 507 : "Mexican vine snake", 
              497 : "King cobra", 495 : "Rough green snake", 485 : "redback coffee snake", 481 : "Brown water snake", 477 : "Florida green watersnake",
              474 : "Mississippi Green Watersnake", 470 : "viperine water snake", 464 : "Cape cobra", 462 : "Indian cobra", 457 : "Chinese cobra",
              454 : "Green tree python Snake", 453 : "Carpet python", 448 : "Texas coral snake", 441 : "Eastern coral snake", 430 : "Striped whipsnake",
              429 : "Schott's whip snake", 427 : "Masticophis lateralis", 424 : "Sonoran whip snake", 422 : "Montpellier snake", 396 : "wolf snake",
              384 : "Mexican parrot snake", 383 : "Pacific Coast parrot snake", 381 : "parrot snake", 373 : "banded cat-eyed snake", 368 : "yellow-lipped sea krait",
              364 : "California mountain kingsnake", 363 : "Milk snake"}

def image_recognition(image):
    print(image)
    resPois = ""
    resCountry = ""
    resScientific = ""
    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, image), result_count=5) # call the classifyImage method of the ImageClassification object to classify the snake in the image
    for eachPrediction, eachProbability in zip(predictions, probabilities): # Iterates through the top 5 predictions in order of decreasing probability
        print(eachPrediction , " : " , eachProbability)
    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, image), result_count=1)
    # print(f"\n\nThis snake is: {snake_dict[int(predictions[0])]}\n\n")
    # print(snake_dict[int(predictions[0])])
    # print (dataSet)

    # running Description code.
    description = snakeDescription(snake_dict[int(predictions[0])])

    #grabbing poisonous value from csv.
    for index,row in dataSet.iterrows():
        poisonous = row['poisonous']
        classValue = row['class_id']
        country = row['country']
        scientific = row['snake_sub_family']
        if classValue == int(predictions[0]):
            if poisonous == 1: 
                resPois = "Venomous"
            else:
                resPois = "Not Venomous"
            resCountry = str(country)
            resScientific = str(scientific)
            break
     
    return [snake_dict[int(predictions[0])], resPois, description, resCountry, scientific]


