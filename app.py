from ui.menu import show_menu
from modes import invisible_object, human_invisible, gravity_vision, mood_camera


def main():
    while True:
        choice = show_menu()

        if choice == "1":
            invisible_object.run()
        elif choice == "2":
            human_invisible.run()
        elif choice == "3":
            gravity_vision.run()
        elif choice == "4":
            mood_camera.run()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()