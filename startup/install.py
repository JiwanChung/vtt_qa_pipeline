import pip


def install_pip():
    if hasattr(pip, 'main'):
        pip.main(['install', '-r', 'requirements.txt'])
    else:
        pip._internal.main(['install', '-r', 'requirements.txt'])


def install_rest():
    import nltk; nltk.download('punkt')


def main():
    install_pip()
    install_rest()


if __name__ == '__main__':
    main()
