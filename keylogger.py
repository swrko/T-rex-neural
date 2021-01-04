from pynput.keyboard import Key, Listener
from pynput import keyboard
from time import sleep


def on_press(key):
    if key == Key.up or key == Key.down:
        print('{0} pressed'.format(
            key))

    if str(key) == "'q'":
        return False


def main():
    # by documentation: runs as thread #non-blocking fashion
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    while True:
        sleep(1)
        print("alive! ")

    # Listener
    # with Listener(on_press=on_press) as listener:
    #     listener.join()


if __name__ == '__main__':
    main()
