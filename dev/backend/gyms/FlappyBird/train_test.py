import gyms.FlappyBird.game.test_flappy as game
import numpy as np
import cv2

def flappytest():
  env = game.GameState()
  action_array = np.zeros(2)
  action_array[0] = 1
  image, reward, terminal = env.frame_step(action_array)
  # 画像を表示
  cv2.imshow('Image', image)
  cv2.waitKey(0)  # キーが押されるまで待機
  cv2.destroyAllWindows()  # ウィンドウを閉じる

if __name__ == '__main__':
  test()