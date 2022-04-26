import cv2
import glob
import numpy as np
from ChessboardFeature import ChessboardFeature
import utlis.ImageProcessing as ImageProcessing
import utlis.SetVariable as SetVariable

workingFolder   = "./images/test"
imageType       = 'jpg'
filename    = workingFolder + "/*." + imageType
images = glob.glob(filename)

ml_data_list = [ [[[781, 292], 'king', 'red'], [[807, 558], 'king', 'green'], [[415, 468], 'pawn', 'green'], [[401, 327], 'pawn', 'red'], [[295, 405], 'bishop', 'red'], [[317, 268], 'bishop', 'green'], [[578, 258], 'rook', 'green'], [[732, 191], 'rook', 'red'], [[511, 647], 'knight', 'red'], [[291, 567], 'knight', 'green'], [[895, 456], 'queen', 'red'], [[598, 464], 'queen', 'green']],
[[[356, 618], 'king', 'green'], [[729, 656], 'king', 'red'], [[510, 328], 'pawn', 'green'], [[689, 335], 'pawn', 'red'], [[766, 275], 'bishop', 'green'], [[590, 253], 'bishop', 'red'], [[919, 651], 'rook', 'red'], [[785, 481], 'rook', 'green'], [[304, 375], 'knight', 'red'], [[414, 241], 'knight', 'green'], [[499, 464], 'queen', 'green'], [[468, 733], 'queen', 'red']],
[[[326, 304], 'king', 'green'], [[273, 564], 'king', 'red'], [[676, 399], 'pawn', 'green'], [[699, 551], 'pawn', 'red'], [[811, 629], 'bishop', 'green'], [[802, 456], 'bishop', 'red'], [[699, 554], 'bishop', 'red'], [[282, 742], 'rook', 'red'], [[497, 634], 'rook', 'green'], [[778, 312], 'knight', 'green'], [[585, 249], 'knight', 'red'], [[212, 384], 'queen', 'red'], [[498, 392], 'queen', 'green']],
[[[370, 251], 'king', 'red'], [[678, 249], 'king', 'green'], [[608, 536], 'pawn', 'green'], [[410, 558], 'pawn', 'red'], [[319, 652], 'bishop', 'green'], [[534, 651], 'bishop', 'red'], [[204, 273], 'rook', 'red'], [[308, 402], 'rook', 'green'], [[804, 451], 'knight', 'red'], [[749, 634], 'knight', 'green'], [[564, 193], 'queen', 'red'], [[586, 390], 'queen', 'green']] ]

color1 = ml_data_list[0][0][2]

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
def pixel_after_warp(p, matrix):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return [int(px), int(py)]

def main():
    for i in range(len(images)):

        ml_data = ml_data_list[i]

        img = cv2.imread(images[i])

        img = ImageProcessing.image_resize(img, height=640)
        color_contrast_img = ImageProcessing.add_contrast(img.copy(), -20, 64)
        color_img = img.copy()
        

        normal_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./images/normal_gray.jpg', normal_gray)

        ImageProcessing.field_contour(img.copy(), './images/clahe_gray.jpg')
        clahe_gray = cv2.imread('./images/clahe_gray.jpg')

        cv2.imwrite("./matlab/Boards/img.jpg", normal_gray)

        corners = SetVariable.Matlab.engine.demo("matlab/Boards/img.jpg", nargout=1)

        chess_corner = []

        for index in range(len(corners[0])):
            chess_corner.append([int(corners[0][index])-1, int(corners[1][index])-1])
        
        # print(chess_corner)
        print('chesscorner: ', np.array(chess_corner).shape)

        result = ""

        if np.array(chess_corner).shape[0] == 81:

            full_chess_corner = chess_corner
            extended_chess_corner = ChessboardFeature.get_extend_edge(full_chess_corner, 9, 1/3)
            # print('full chesscorner: ', np.array(full_chess_corner).shape)

            # Displaying the image
            detected_chessboard = cv2.imread('./images/normal_gray.jpg')

            for i in extended_chess_corner:
                detected_chessboard = cv2.circle(detected_chessboard, (int(i[0]), int(i[1])), 3, (0, 0, 255), -1)
            cv2.imwrite('./images/Detect.jpg', detected_chessboard)
            warp_corners = ChessboardFeature.get_warp_corner(full_chess_corner, extended_chess_corner, cv2.cvtColor(clahe_gray.copy(), cv2.COLOR_BGR2GRAY))

            new_contrast_img = ImageProcessing.add_contrast(img.copy(), 40, 40)
            warped_chessboard, matrix = ImageProcessing.warp_corner(warp_corners, cv2.cvtColor(new_contrast_img.copy(), cv2.COLOR_BGR2GRAY), 640, 640)
            # warped_chessboard = ImageProcessing.warp_corner(warp_corners, normal_gray.copy(), 640, 640)

            warped_color_img, matrix = ImageProcessing.warp_corner(warp_corners, color_img, 640, 640)
            warped_color_contrast_img, matrix = ImageProcessing.warp_corner(warp_corners, color_contrast_img, 640, 640)

            
            mask = np.zeros([640,640], dtype="uint8")


            imageType       = 'jpg'
            filename    = "./images/color_img" + "/*." + imageType
            fname      = glob.glob(filename)


            for i in range(len(fname)-1, -1, -1):
                
                if fname[i][len(fname[i])-6] in ['a','c','e','g'] and fname[i][len(fname[i])-5] in ['1', '3', '5', '7']:
                    board_color = 'black.jpg'
                    font_color = (255,255,255)
                elif fname[i][len(fname[i])-6] in ['b', 'd', 'f', 'h'] and fname[i][len(fname[i])-5] in ['2', '4', '6', '8']:
                    board_color = 'black.jpg'
                    font_color = (255,255,255)
                else:
                    board_color = 'white.jpg'
                    font_color = (0,0,0)
                
                final_img = cv2.imread('./images/board_template/'+board_color)
                final_img = cv2.imread('./images/board_template/'+board_color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_img, fname[i][len(fname[i])-6]+fname[i][len(fname[i])-5], (5,15), font, 0.4, font_color, 1, cv2.LINE_AA)
                cv2.imwrite('images/final_img' + '/' + fname[i][len(fname[i])-6::], final_img)

            
            for index, data in enumerate(ml_data):
                if data[2] == color1:
                    ml_data[index][2] = 0
                else:
                    ml_data[index][2] = 1

                new_point = pixel_after_warp([data[0][0]*640/1080, data[0][1]*640/1080], matrix)
                # print(data[0], new_point)
                ml_data[index][0] = new_point
                mask = cv2.circle(mask, (new_point[0], new_point[1]), 15, (255, 255, 255), -1)

                if (int(new_point[1])) % 640 != 0:
                    y = (new_point[1]//80) + 1
                else:
                    y = (new_point[1]//80) 
                
                if (int(new_point[0])) % 640 == 0:
                    x = (new_point[0]//80) + 1
                else:
                    x = (new_point[0]//80) 
                
                y = 8-y
                
                val_num = (x + ((y-1) * 8))
                ml_data[index].append(alphabet[x] + str(y+1))

                if alphabet[x] in ['a','c','e','g'] and str(y+1) in ['1', '3', '5', '7']:
                    board_color = 'black.jpg'
                    font_color = (255,255,255)
                elif alphabet[x] in ['b', 'd', 'f', 'h'] and str(y+1) in ['2', '4', '6', '8']:
                    board_color = 'black.jpg'
                    font_color = (255,255,255)
                else:
                    board_color = 'white.jpg'
                    font_color = (0,0,0)

                final_img = cv2.imread('./images/board_template/'+board_color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_img, alphabet[x]+str(y+1), (5,15), font, 0.4, font_color, 1, cv2.LINE_AA)
                
                if  ml_data[index][2] == 0:
                    # cv2.circle(final_img,(40,40), 15, (0,0,255), -1)
                    cname = 'w_'
                elif  ml_data[index][2] == 1:
                    # cv2.circle(final_img,(40,40), 15, (0,255,0), -1)
                    cname = 'b_'
                
                cname += (ml_data[index][1] + '.png')

                cfile = cv2.imread("./images/chess_template/" + cname, -1)

                # print("./images/chess_template/" + cname)


                cfile = ImageProcessing.image_resize(cfile, height=50)

                x_offset = (80 - cfile.shape[1])//2
                y_offset = (80 - cfile.shape[0])//2

                y1, y2 = y_offset, y_offset + cfile.shape[0]
                x1, x2 = x_offset, x_offset + cfile.shape[1]

                alpha_s = cfile[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                
                for c in range(0, 3):
                    final_img[y1:y2, x1:x2, c] = (alpha_s * cfile[:, :, c] + alpha_l * final_img[y1:y2, x1:x2, c])

        
                cv2.imwrite('images/final_img/' + str(alphabet[x]) + str(y+1) +'.jpg', final_img)
                
            imageType       = 'jpg'
            filename    = "./images/final_img" + "/*." + imageType
            fname      = glob.glob(filename)


            horz_final = []
            vert_final = [[],[],[],[],[],[],[],[]]

            row = 7

            for i in range(len(fname)-1, -1, -1):
                final_img     = cv2.imread(fname[i])
                horz_final.append(final_img)

                if i != 64 and (i)%8 == 0:
                    H1 = np.concatenate(horz_final, axis=0)
                    vert_final[row] = H1
                    horz_final = []
                    row -= 1

            final_img = np.concatenate(vert_final, axis=1)
            cv2.imwrite('images/final_img.jpg', final_img)
            # cv2.imshow("mask", mask)
                

            cv2.imshow('Final', final_img)

            for index, info in enumerate(ml_data):
                for alp_index, letter in enumerate(alphabet):
                    if info[3][0] == letter:
                        cell_num = ((alp_index) + ((int(info[3][1])-1) * 8))
                        
                result += str(info[1])+','+ str(info[2])+',' + str(cell_num) +':'
            print (result)

        key = cv2.waitKey()
        if key == 27:
            break
        
if __name__ == "__main__":
    main() 