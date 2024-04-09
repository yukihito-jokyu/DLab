import numpy as np
import sys
import random
import pygame
import gyms.FlappyBird.game.flappy_bird_utils as flappy_bird_utils
from itertools import cycle

class GameState:
    def __init__(self):
        self.FPS = 30
        self.PIPEGAPSIZE = 100 
        self.BASEY = 512 * 0.79
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = 50
        self.playery = 512 * 0.5
        self.basex = 0
        self.baseShift = 288 - 336

        self.PLAYER_WIDTH = 34
        self.PLAYER_HEIGHT = 24
        self.PIPE_WIDTH = 52
        self.PIPE_HEIGHT = 320
        self.BACKGROUND_WIDTH = 288

        self.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

        self.pipeVelX = -4
        self.playerVelY = 0
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -9
        self.playerFlapped = False

        self.upperPipes = []
        self.lowerPipes = []

        # Pygameの初期化
        pygame.init()

    def frame_step(self, input_actions):
        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        if input_actions[1] == 1:
            if self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        if 30 < playerMidPos < 90:
            self.score += 1
            reward = 1

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 4) % self.baseShift)

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        if self.upperPipes and self.lowerPipes and self.pipeVelX < 0 and self.upperPipes[0]['x'] < -self.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        if self.upperPipes and self.lowerPipes and 0 < self.upperPipes[0]['x'] < 5:
            new_pipe = self.getRandomPipe()
            self.upperPipes.append(new_pipe[0])
            self.lowerPipes.append(new_pipe[1])

        isCrash = self.checkCrash()
        if isCrash:
            terminal = True
            reward = -1

        state_image = self.get_screen_image()
        # state = [self.playery, self.upperPipes[0]['y'], self.lowerPipes[0]['y']]
        return state_image, reward, terminal

    def getRandomPipe(self):
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
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

    def get_screen_image(self):
        # 背景を作成
        background = pygame.Surface((288, 512))

        # 背景を塗りつぶし
        background.fill((0, 0, 0))

        # パイプを描画
        for u_pipe, l_pipe in zip(self.upperPipes, self.lowerPipes):
            pygame.draw.rect(background, (255, 255, 255), pygame.Rect(u_pipe['x'], 0, self.PIPE_WIDTH, u_pipe['y']))
            pygame.draw.rect(background, (255, 255, 255), pygame.Rect(l_pipe['x'], l_pipe['y'] + self.PIPEGAPSIZE, self.PIPE_WIDTH, 512 - (l_pipe['y'] + self.PIPEGAPSIZE)))

        # プレイヤーを描画
        pygame.draw.rect(background, (255, 255, 255), pygame.Rect(self.playerx, self.playery, self.PLAYER_WIDTH, self.PLAYER_HEIGHT))

        # pygame.surface.Surfaceをnumpy.ndarrayに変換して返す
        return pygame.surfarray.array3d(background)

    def close(self):
        pygame.quit()
