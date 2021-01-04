from pynput import keyboard
from pynput.keyboard import Key, Listener

def main():
    def on_press(key):
        if key == Key.up or key == Key.down:
            print('{0} pressed'.format(
                key))
        # if key == Key.esc:
            # Stop listene
        if str(key) == "'q'":
            return False

    # Collect events until released
    with Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    main()


# def main():
#     with keyboard.Events() as events:
#         for event in events:
#             if event.key ==:
#                 break
#             else:
#                 print("pressed key: {}".format(event))
