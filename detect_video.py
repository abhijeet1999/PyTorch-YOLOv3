from __future__ import division
from torch.autograd import Variable
import cv2 
from utill.img_process import inp_to_image, resize_img
from utill.model_video import *
from utill.util import *
import pandas as pd
import random 
import pickle as pkl
import argparse


def prepare_input(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    Perform tranpose and return Tensor
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (resize_img(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
    
def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img



def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "config/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile and you have to specify ur weight file path",
                        default = "trainedmodel.pth", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "316", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    if torch.cuda.is_available():
        CUDA=lambda storage, loc: storage.cuda()
    else:
        CUDA='cpu'
    


    #CUDA = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 20
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network")
    model = Darknet(args.cfgfile).to(CUDA)
    model.load_state_dict(torch.load(args.weightsfile,map_location= CUDA ))
    print("Network loaded")
    classes = load_classes('data/custom/classes.names') # specify  the path of your class names
    colors = pkl.load(open("pallete", "rb"))
    model.hyperparams["height"] = args.reso
    inp_dim = int(model.hyperparams["height"])
    print("cuda",CUDA)


    
        
    model.eval()
    
    videofile = args.video
    #well this made for web cam for now thats why it is 0
    cap = cv2.VideoCapture(0) #cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            

            img, orig_im, dim = prepare_input(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            with torch.no_grad(): 
                output = model(Variable(img))
                
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('x'):
                    break
                continue
            
            
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            

            
            list(map(lambda x: write(x, orig_im), output))
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'): # press x to exit
                break
        else:
            break
    

    
    

