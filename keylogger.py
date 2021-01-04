from pynput.keyboard import Key, Listener


def main():
    def on_press(key):
        if key == Key.up or key == Key.down:
            print('{0} pressed'.format(
                key))

        if str(key) == "'q'":
            return False

    # Listener
    with Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    main()
