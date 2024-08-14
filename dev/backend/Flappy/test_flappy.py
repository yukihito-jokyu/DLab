import numpy as np
import cv2
import random
import pygame
from itertools import cycle

class GameState:
    def __init__(self):
        self.lPipe_image = cv2.cvtColor(cv2.imread('Flappy/assets/pipe-green.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2BGR)
        self.Base_image = cv2.cvtColor(cv2.imread('Flappy/assets/base.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2BGR)
        self.uPipe_image = cv2.cvtColor(cv2.rotate(self.lPipe_image, cv2.ROTATE_180), cv2.COLOR_BGRA2BGR)
        self.player_down_image = cv2.cvtColor(cv2.imread('Flappy/assets/redbird-downflap.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2BGR)
        self.player_up_image = cv2.cvtColor(cv2.imread('Flappy/assets/redbird-upflap.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2BGR)
        print(f'写真{self.lPipe_image.shape}')
        print(f'写真{self.uPipe_image.shape}')
        print(self.uPipe_image[320-30:320, :].shape)
        self.PIPEGAPSIZE = 100 
        self.BASEY = 512 * 0.79
        self.playerx = int(512*0.2)
        self.playery = int((512-24)/2)
        self.basex = 0
        self.baseShift = 288 - 336

        self.PLAYER_WIDTH = 34
        self.PLAYER_HEIGHT = 24
        self.PIPE_WIDTH = 52
        self.PIPE_HEIGHT = 320
        self.BACKGROUND_WIDTH = 288

        self.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

        self.score = self.playerIndex = self.loopIter = 0
        self.pipeVelX = -4
        self.playerVelY = 0
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        self.upperPipes = [
            {'x': 288, 'y': newPipe1[0]['y']},
            {'x': 288 + (288 / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': 288, 'y': newPipe1[1]['y']},
            {'x': 288 + (288 / 2), 'y': newPipe2[1]['y']},
        ]

        self.background = np.zeros((512, 288, 3), dtype=np.uint8)

        self.scroll_offset = 0
        self.scroll_speed = 3  # スクロール速度（必要に応じて調整）
        self.action = 0

        # Pygameの初期化
        pygame.init()

    def frame_step(self, input_actions):
        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        
        if input_actions[0] == 1:
            self.action = 0
        else:
            self.action = 1

        if input_actions[1] == 1:
            if self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        if self.upperPipes[0]['x'] < -self.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        isCrash = self.checkCrash()
        # isCrash = False
        if isCrash:
            terminal = True
            self.__init__()
            reward = -1

        self.make_screen_image()
        return self.background, reward, terminal

    def getRandomPipe(self):
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs)-1)
        gapY = gapYs[index]

        gapY += int(self.BASEY * 0.2)
        pipeX = 288 + 10
        return [
            {'x': pipeX, 'y': gapY - self.PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(self):
        if self.playery + self.PLAYER_HEIGHT >= self.BASEY - 1:
            return True
        player_rect = pygame.Rect(self.playerx, self.playery, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        for u_pipe, l_pipe in zip(self.upperPipes, self.lowerPipes):
            u_pipe_rect = pygame.Rect(u_pipe['x'], u_pipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)
            l_pipe_rect = pygame.Rect(l_pipe['x'], l_pipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)
            if player_rect.colliderect(u_pipe_rect) or player_rect.colliderect(l_pipe_rect):
                return True
        return False
    
    def get_low_broadcast_index(self, x, y):
        # 左から右
        back_broad_x = [0, 0]
        image_broad_x = [0, 0]
        if x+self.PIPE_WIDTH > 288:
            back_broad_x[0] = x
            back_broad_x[1] = 288
            image_broad_x[0] = 0
            image_broad_x[1] = 288-x
        elif x < 0:
            back_broad_x[0] = 0
            back_broad_x[1] = self.PIPE_WIDTH-x
            image_broad_x[0] = 0 - x
            image_broad_x[1] = self.PIPE_WIDTH
        else:
            back_broad_x[0] = x
            back_broad_x[1] = x+self.PIPE_WIDTH
            image_broad_x[0] = 0
            image_broad_x[1] = self.PIPE_WIDTH
        # 上から下
        back_broad_y = [0, 0]
        back_broad_y[0] = y
        back_broad_y[1] = 512
        return back_broad_x, back_broad_y

    def make_screen_image(self):
        # 背景を作成（黒で塗りつぶし）
        self.background = np.zeros((512, 288, 3), dtype=np.uint8)
        
        # パイプを描画
        # for u_pipe, l_pipe in zip(self.upperPipes, self.lowerPipes):
        #     l_x, l_y = l_pipe['x'], l_pipe['y']
        #     print(f'l_x{l_x}, l_y{l_y}')
        #     # l_broad_x, l_broad_y = self.get_low_broadcast_index(l_x, l_y)
        #     # background[l_y:512, l_x:l_x+self.lPipe_image.shape[1], :] = self.lPipe_image
        #     u_pipe_x, u_pipe_y = int(u_pipe['x']), int(u_pipe['y'])
        #     l_pipe_x, l_pipe_y = int(l_pipe['x']), int(l_pipe['y'] + self.PIPEGAPSIZE)
        #     print(f'上土管の左上{(u_pipe_x, 0)}　右上{(u_pipe_x + self.PIPE_WIDTH, u_pipe_y + self.PIPE_HEIGHT)}')
        #     print(f'下土管の左上{(l_pipe_x, l_pipe_y+112)}　右上{(l_pipe_x + self.PIPE_WIDTH, 512)}')
        #     cv2.rectangle(self.background, (u_pipe_x, 0), (u_pipe_x + self.PIPE_WIDTH, u_pipe_y + self.PIPE_HEIGHT), (255, 0, 0), -1)
        #     cv2.rectangle(self.background, (l_pipe_x, l_pipe_y-112), (l_pipe_x + self.PIPE_WIDTH, 512-112), (0, 255, 0), -1)
        # パイプを描画
        # for u_pipe, l_pipe in zip(self.upperPipes, self.lowerPipes):
        #     u_pipe_x, u_pipe_y = int(u_pipe['x']), int(u_pipe['y'])
        #     l_pipe_x, l_pipe_y = int(l_pipe['x']), int(l_pipe['y'] + self.PIPEGAPSIZE)

        #     # 上パイプの描画
        #     pipe_height = u_pipe_y + self.PIPE_HEIGHT
        #     if pipe_height > 0:
        #         crop_height = min(320, pipe_height)
        #         crop_width = min(52, self.BACKGROUND_WIDTH - u_pipe_x)
        #         if crop_width > 0:
        #             self.background[0:crop_height, u_pipe_x:u_pipe_x+crop_width] = self.uPipe_image[320-crop_height:320, :crop_width]

        #     # 下パイプの描画
        #     pipe_start_y = l_pipe_y - 112
        #     if pipe_start_y < 400:  # 512 - 112 = 400（地面の開始位置）
        #         crop_start = max(0, -pipe_start_y)
        #         crop_end = min(320, 400 - pipe_start_y)
        #         bg_start_y = max(0, pipe_start_y)
        #         crop_width = min(52, self.BACKGROUND_WIDTH - l_pipe_x)
        #         if crop_width > 0:
        #             self.background[bg_start_y:400, l_pipe_x:l_pipe_x+crop_width] = self.lPipe_image[crop_start:crop_end, :crop_width]

        #     print(f'上土管の左上{(u_pipe_x, 0)}　右上{(u_pipe_x + self.PIPE_WIDTH, u_pipe_y + self.PIPE_HEIGHT)}')
        #     print(f'下土管の左上{(l_pipe_x, l_pipe_y+112)}　右上{(l_pipe_x + self.PIPE_WIDTH, 512)}')

        
        # パイプを描画
        # for u_pipe, l_pipe in zip(self.upperPipes, self.lowerPipes):
        #     l_x, l_y = l_pipe['x'], l_pipe['y']
        #     print(f'l_x{l_x}, l_y{l_y}')
        #     # l_broad_x, l_broad_y = self.get_low_broadcast_index(l_x, l_y)
        #     # background[l_y:512, l_x:l_x+self.lPipe_image.shape[1], :] = self.lPipe_image
        #     u_pipe_x, u_pipe_y = int(u_pipe['x']), int(u_pipe['y'])
        #     l_pipe_x, l_pipe_y = int(l_pipe['x']), int(l_pipe['y'] + self.PIPEGAPSIZE)
        #     print(f'上土管の左上{(u_pipe_x, 0)}　右上{(u_pipe_x + self.PIPE_WIDTH, u_pipe_y + self.PIPE_HEIGHT)}')
        #     print(f'下土管の左上{(l_pipe_x, l_pipe_y+112)}　右上{(l_pipe_x + self.PIPE_WIDTH, 512)}')
        #     cv2.rectangle(self.background, (u_pipe_x, 0), (u_pipe_x + self.PIPE_WIDTH, u_pipe_y + self.PIPE_HEIGHT), (255, 0, 0), -1)
        #     cv2.rectangle(self.background, (l_pipe_x, l_pipe_y-112), (l_pipe_x + self.PIPE_WIDTH, 512-112), (0, 255, 0), -1)
        self.make_pipe()
        # 地面
        # cv2.rectangle(self.background, (0, 512-112), (288, 512), (255, 0, 255), -1)
        self.make_base()
        # プレイヤーを描画
        player_x, player_y = int(self.playerx), int(self.playery)
        self.make_player()
        # cv2.rectangle(self.background, (player_x, player_y), (player_x + self.PLAYER_WIDTH, player_y + self.PLAYER_HEIGHT), (0, 0, 255), -1)
    
    def make_player(self):
        player_x, player_y = int(self.playerx), int(self.playery)
        # プレイヤー画像を背景の指定エリアに貼り付け
        x_start, y_start = player_x, player_y
        x_end, y_end = x_start + self.PLAYER_WIDTH, y_start + self.PLAYER_HEIGHT

        # プレイヤー画像を指定エリアに合わせてリサイズ
        player_up_image_resized = cv2.resize(self.player_up_image, (x_end - x_start, y_end - y_start))
        player_down_image_resized = cv2.resize(self.player_down_image, (x_end - x_start, y_end - y_start))

        # 背景画像にプレイヤー画像を貼り付け
        if self.action == 0:
            self.background[y_start:y_end, x_start:x_end] = player_down_image_resized
        else:
            self.background[y_start:y_end, x_start:x_end] = player_up_image_resized

    def make_pipe(self):
        for u_pipe, l_pipe in zip(self.upperPipes, self.lowerPipes):
            u_pipe_x, u_pipe_y = int(u_pipe['x']), int(u_pipe['y'])
            l_pipe_x, l_pipe_y = int(l_pipe['x']), int(l_pipe['y'] + self.PIPEGAPSIZE)

            # 上パイプの描画
            pipe_height = u_pipe_y + self.PIPE_HEIGHT
            if pipe_height > 0:
                crop_height = min(320, pipe_height)
                crop_width = min(52, self.BACKGROUND_WIDTH - u_pipe_x)
                if crop_width > 0:
                    pipe_image = self.uPipe_image[320-crop_height:320, :crop_width].copy()
                    # 白い部分を黒に変更
                    white_mask = np.all(pipe_image == [255, 255, 255], axis=-1)
                    pipe_image[white_mask] = [0, 0, 0]
                    self.background[0:crop_height, u_pipe_x:u_pipe_x+crop_width] = pipe_image

            # 下パイプの描画
            pipe_start_y = l_pipe_y - 112
            if pipe_start_y < 400:  # 512 - 112 = 400（地面の開始位置）
                crop_start = max(0, -pipe_start_y)
                crop_end = min(320, 400 - pipe_start_y)
                bg_start_y = max(0, pipe_start_y)
                crop_width = min(52, self.BACKGROUND_WIDTH - l_pipe_x)
                if crop_width > 0:
                    pipe_image = self.lPipe_image[crop_start:crop_end, :crop_width].copy()
                    # 白い部分を黒に変更
                    white_mask = np.all(pipe_image == [255, 255, 255], axis=-1)
                    pipe_image[white_mask] = [0, 0, 0]
                    self.background[bg_start_y:400, l_pipe_x:l_pipe_x+crop_width] = pipe_image

            
    def make_base(self):
        # 地面のエリアを描画
        cv2.rectangle(self.background, (0, 512-112), (288, 512), (255, 0, 255), -1)

        # 地面のエリアを描画（スクロールのため、幅を2倍に）
        ground_width = self.BACKGROUND_WIDTH * 2
        ground_height = 112
        ground = np.zeros((ground_height, ground_width, 3), dtype=np.uint8)

        # PNG画像を地面の高さに合わせてリサイズ
        overlay_image_resized = cv2.resize(self.Base_image, (self.Base_image.shape[1], ground_height))

        # アルファチャンネルがある場合は除去
        if overlay_image_resized.shape[2] == 4:
            overlay_image_resized = overlay_image_resized[:, :, :3]

        # 画像を地面に繰り返し配置
        for i in range(0, ground_width, overlay_image_resized.shape[1]):
            if i + overlay_image_resized.shape[1] > ground_width:
                width = ground_width - i
                ground[:, i:] = overlay_image_resized[:, :width]
            else:
                ground[:, i:i+overlay_image_resized.shape[1]] = overlay_image_resized

        # スクロール位置を更新（クラスの属性として scroll_offset を追加する必要があります）
        self.scroll_offset = (self.scroll_offset + self.scroll_speed) % self.BACKGROUND_WIDTH  # スクロール速度は2ピクセルとします

        # スクロールした地面を背景に配置
        self.background[512-112:512, :] = ground[:, self.scroll_offset:self.scroll_offset+self.BACKGROUND_WIDTH]


        # # PNG画像のサイズ
        # overlay_height, overlay_width = self.Base_image.shape[:2]

        # # 背景画像の指定エリアの座標
        # x_start, y_start = 0, 512 - 112
        # x_end, y_end = x_start + 288, y_start + 112

        # # PNG画像を背景画像の指定エリアに合わせてリサイズ
        # overlay_image_resized = cv2.resize(self.Base_image, (x_end - x_start, y_end - y_start))

        # # 画像のサイズを確認
        # print(f"Resized overlay image shape: {overlay_image_resized.shape}")

        # # PNG画像が透過を含む場合、アルファチャンネルを利用して貼り付ける
        # if overlay_image_resized.shape[2] == 4:  # 透過情報がある場合
        #     alpha_channel = overlay_image_resized[:, :, 3] / 255.0
        #     for c in range(3):  # RGB各チャンネルについて
        #         self.background[y_start:y_end, x_start:x_end, c] = (
        #             (1 - alpha_channel) * self.background[y_start:y_end, x_start:x_end, c] +
        #             alpha_channel * overlay_image_resized[:, :, c]
        #         )
        # else:
        #     self.background[y_start:y_end, x_start:x_end] = overlay_image_resized


    def close(self):
        pygame.quit()
