# English

# How to Install and Run PyCharm 2024.1.4

This guide outlines the steps to install and run PyCharm 2024.1.4 on an Ubuntu server. It includes all necessary steps to use PyCharm in a GUI environment via X11 forwarding using PuTTY.

---

## 1. Download and Install PyCharm

1. **Download PyCharm**:
    ```bash
    wget https://download.jetbrains.com/python/pycharm-community-2024.1.4.tar.gz
    ```
    (Replace the URL with the latest version from the [PyCharm download page](https://www.jetbrains.com/pycharm/download/#section=linux).)

2. **Extract the downloaded file**:
    ```bash
    tar -xzf pycharm-community-2024.1.4.tar.gz
    ```

3. **Navigate to the extracted directory**:
    ```bash
    cd pycharm-community-2024.1.4/bin
    ```

4. **Run PyCharm**:
    ```bash
    ./pycharm.sh
    ```

---

## 2. Create a PyCharm Run Script

1. **Create a shell script**:
    ```bash
    nano ~/run_pycharm.sh
    ```

2. **Add the following content**:
    ```bash
    #!/bin/bash
    /home/usr/pycharm-2024.1.4/bin/pycharm.sh
    ```

3. **Save and close the file**:
    - Press `Ctrl + O` to save, and `Ctrl + X` to exit.

4. **Make the script executable**:
    ```bash
    chmod +x ~/run_pycharm.sh
    ```

5. **Run PyCharm using the script**:
    ```bash
    ~/run_pycharm.sh
    ```

---

## 3. Create a Desktop Entry File (Optional)

### Create and Edit the Desktop Entry File

1. **Create the desktop entry file**:
    ```bash
    nano ~/pycharm.desktop
    ```

2. **Add the following content**:
    ```plaintext
    [Desktop Entry]
    Version=1.0
    Type=Application
    Name=PyCharm
    Icon=/home/usr/pycharm-2024.1.4/bin/pycharm.png
    Exec="/home/usr/pycharm-2024.1.4/bin/pycharm.sh" %f
    Comment=Python IDE
    Categories=Development;IDE;
    Terminal=false
    ```

3. **Save and close the file**:
    - Press `Ctrl + O` to save, and `Ctrl + X` to exit.

4. **Make the desktop entry file executable**:
    ```bash
    chmod +x ~/pycharm.desktop
    ```

5. **Find and run the desktop entry file using the file explorer**:
    - Open the file explorer in the home directory and double-click the `pycharm.desktop` file.
    - If it doesn't run, right-click and select "Open With" and choose the appropriate application.

This should make it easy to run PyCharm. If you have any further questions, feel free to ask.

# Korean

# PyCharm 2024.1.4 설치 및 실행 방법

Ubuntu 서버에 PyCharm 2024.1.4 버전을 설치하고 실행하는 방법을 정리하였습니다. 이 가이드는 PuTTY를 통해 X11 포워딩을 사용하여 PyCharm을 GUI 환경에서 실행하는 데 필요한 모든 단계를 포함합니다.

---

## 1. PyCharm 다운로드 및 설치

1. **PyCharm 다운로드**:
    ```bash
    wget https://download.jetbrains.com/python/pycharm-community-2024.1.4.tar.gz
    ```
    (URL은 최신 버전으로 대체해야 합니다. [PyCharm 다운로드 페이지](https://www.jetbrains.com/pycharm/download/#section=linux)에서 최신 링크를 확인하세요.)

2. **다운로드한 파일 추출**:
    ```bash
    tar -xzf pycharm-community-2024.1.4.tar.gz
    ```

3. **추출된 디렉토리로 이동**:
    ```bash
    cd pycharm-community-2024.1.4/bin
    ```

4. **PyCharm 실행**:
    ```bash
    ./pycharm.sh
    ```

---

## 2. PyCharm 실행 스크립트 생성

1. **쉘 스크립트 생성**:
    ```bash
    nano ~/run_pycharm.sh
    ```

2. **다음 내용을 추가**:
    ```bash
    #!/bin/bash
    /home/usr/pycharm-2024.1.4/bin/pycharm.sh
    ```

3. **파일 저장 및 닫기**:
    - `Ctrl + O`를 눌러 저장하고, `Ctrl + X`를 눌러 나옵니다.

4. **스크립트 파일에 실행 권한 부여**:
    ```bash
    chmod +x ~/run_pycharm.sh
    ```

5. **PyCharm 실행**:
    ```bash
    ~/run_pycharm.sh
    ```

---

## 3. 데스크탑 파일 생성 (선택 사항)

### 데스크탑 파일 생성 및 편집

1. **데스크탑 파일 생성**:
    ```bash
    nano ~/pycharm.desktop
    ```

2. **다음 내용을 추가**:
    ```plaintext
    [Desktop Entry]
    Version=1.0
    Type=Application
    Name=PyCharm
    Icon=/home/usr/pycharm-2024.1.4/bin/pycharm.png
    Exec="/home/usr/pycharm-2024.1.4/bin/pycharm.sh" %f
    Comment=Python IDE
    Categories=Development;IDE;
    Terminal=false
    ```

3. **파일 저장 및 닫기**:
    - `Ctrl + O`를 눌러 저장하고, `Ctrl + X`를 눌러 나옵니다.

4. **데스크탑 파일에 실행 권한 부여**:
    ```bash
    chmod +x ~/pycharm.desktop
    ```

5. **파일 탐색기에서 데스크탑 파일을 찾아서 실행**:
    - 홈 디렉토리에서 파일 탐색기를 열고 `pycharm.desktop` 파일을 더블 클릭하여 실행합니다.
    - 만약 파일 탐색기에서 실행이 되지 않는다면, 우클릭 후 "프로그램으로 열기"를 선택합니다.

이제 PyCharm을 쉽게 실행할 수 있습니다. 추가로 궁금한 사항이 있으면 언제든지 말씀해 주세요.
