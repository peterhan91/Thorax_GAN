import cxr_dataset as CXR
import eval_model as E
import model as M


# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "C:/NIH/images"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
if __name__ == '__main__':
    preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)

