import torch
import numpy as np
import cv2
import ref

public class ChromaToTransparency {

    public ChromaToTransparency(){
        MarvinImage image = MarvinImageIO.loadImage("./res/person_chroma.jpg");
        MarvinImage imageOut = new MarvinImage(image.getWidth(), image.getHeight());

        // 1. Convert green to transparency
        greenToTransparency(image, imageOut);
        MarvinImageIO.saveImage(imageOut, "./res/person_chroma_out1.png");

        // 2. Reduce remaining green pixels
        reduceGreen(imageOut);
        MarvinImageIO.saveImage(imageOut, "./res/person_chroma_out2.png");

        // 3. Apply alpha to the boundary
        alphaBoundary(imageOut, 6);
        MarvinImageIO.saveImage(imageOut, "./res/person_chroma_out3.png");

    }

    private void greenToTransparency(MarvinImage imageIn, MarvinImage imageOut){
        for(int y=0; y<imageIn.getHeight(); y++){
            for(int x=0; x<imageIn.getWidth(); x++){

                int color = imageIn.getIntColor(x, y);
                int r = imageIn.getIntComponent0(x, y);
                int g = imageIn.getIntComponent1(x, y);
                int b = imageIn.getIntComponent2(x, y);

                double[] hsv = MarvinColorModelConverter.rgbToHsv(new int[]{color});

                if(hsv[0] >= 60 && hsv[0] <= 130 && hsv[1] >= 0.4 && hsv[2] >= 0.3){
                    imageOut.setIntColor(x, y, 0, 127, 127, 127);
                }
                else{
                    imageOut.setIntColor(x, y, color);
                }

            }
        }
    }

    private void reduceGreen(MarvinImage image){
        for(int y=0; y<image.getHeight(); y++){
            for(int x=0; x<image.getWidth(); x++){
                int r = image.getIntComponent0(x, y);
                int g = image.getIntComponent1(x, y);
                int b = image.getIntComponent2(x, y);
                int color = image.getIntColor(x, y);
                double[] hsv = MarvinColorModelConverter.rgbToHsv(new int[]{color});

                if(hsv[0] >= 60 && hsv[0] <= 130 && hsv[1] >= 0.15 && hsv[2] > 0.15){
                    if((r*b) !=0 && (g*g) / (r*b) >= 1.5){
                        image.setIntColor(x, y, 255, (int)(r*1.4), (int)g, (int)(b*1.4));
                    } else{
                        image.setIntColor(x, y, 255, (int)(r*1.2), g, (int)(b*1.2));
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        new ChromaToTransparency();
    }
}
