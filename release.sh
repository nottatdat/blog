commit_all() {
    git add -A
    git commit -m "$1"
    git push -f origin master
}

release_blog() {
    hugo
    cd public
    commit_all "$1"
    cd ..
}

release() {
    release_blog "$1"
    commit_all "$1"
}

release "$1"
