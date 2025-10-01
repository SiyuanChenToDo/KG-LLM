### GitHub 项目管理常用指令大全

这份文档涵盖了从初始化项目到日常更新维护的全部核心指令。

#### 一、 初始设置 (只需配置一次)

这些指令用于首次在您的电脑上配置 Git。

```bash
# 设置您的 GitHub 用户名
git config --global user.name ""

# 设置您的 GitHub 邮箱地址
git config --global user.email ""
```

---

#### 二、 创建与上传新项目

当您有一个本地项目，想把它上传到 GitHub 时，按以下顺序操作。

1.  **进入项目文件夹**
    ```bash
    cd /path/to/your/project
    ```

2.  **初始化本地 Git 仓库**
    ```bash
    git init
    ```

3.  **将所有文件添加到暂存区**
    ```bash
    git add .
    ```

4.  **提交文件到本地仓库**
    ```bash
    git commit -m "Initial commit"
    ```

5.  **在 GitHub.com 上创建一个新的空仓库** (不要勾选初始化选项)

6.  **关联本地仓库与远程仓库** (将 URL 替换为您自己仓库的地址)
    ```bash
    git remote add origin https://github.com/SiyuanChenToDo/MyFirstProject.git
    ```

7.  **推送代码到 GitHub**
    ```bash
    git push -u origin master
    ```
    *   `-u` 参数会记住 `origin` 和 `master` 的关联，未来推送时可以直接使用 `git push`。
    *   **注意**: 较新的 Git 版本可能使用 `main` 作为默认分支名，而不是 `master`。如果推送失败，请检查并替换为 `main`。

---

#### 三、 日常更新与修改

当您在本地修改了代码，需要将改动同步到 GitHub 时。

1.  **查看文件状态** (可选，查看哪些文件被修改了)
    ```bash
    git status
    ```

2.  **添加改动到暂存区**
    ```bash
    # 添加所有被修改和新建的文件
    git add .
    
    # 或者只添加某个特定的文件
    git add path/to/your/file.py
    ```

3.  **提交改动到本地仓库** (写清楚本次提交做了什么修改)
    ```bash
    git commit -m "一个清晰的提交信息, 例如: 修复了用户登录的 bug"
    ```

4.  **推送更新到 GitHub**
    ```bash
    git push
    ```

---

#### 四、 同步与拉取更新

当别人修改了代码并推送到了 GitHub，您需要将这些更新同步到您的本地。

1.  **从远程仓库拉取最新代码**
    ```bash
    git pull
    ```
    *   `git pull` 实际上是 `git fetch` (抓取更新) 和 `git merge` (合并更新) 两个命令的组合。
    *   在修改本地代码前，先执行 `git pull` 是一个好习惯，可以避免很多冲突。

---

#### 五、 分支管理

分支允许多人同时在项目的不同部分工作，而不会互相干扰。

1.  **查看所有分支**
    ```bash
    git branch
    ```

2.  **创建一个新分支**
    ```bash
    git branch new-feature-branch
    ```

3.  **切换到一个分支**
    ```bash
    git checkout new-feature-branch
    ```
    *   或者，**创建并直接切换**到一个新分支：
        ```bash
        git checkout -b new-feature-branch
        ```

4.  **将新分支推送到 GitHub** (首次推送新分支时需要)
    ```bash
    git push --set-upstream origin new-feature-branch
    ```

5.  **合并分支**
    当您在新分支上的功能开发完成后，需要把它合并回主分支 (例如 `master` 或 `main`)。
    ```bash
    # 首先，切换回主分支
    git checkout master
    
    # 拉取最新的主分支代码，确保它是最新的
    git pull
    
    # 将新分支合并到当前的主分支
    git merge new-feature-branch
    ```

6.  **删除一个本地分支**
    ```bash
    git branch -d new-feature-branch
    ```
