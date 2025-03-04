git clean -fx
rm -rf build
rm -rf pymsz.egg-info
rm -rf pymsz/*.pyc
rm -rf dist
cd pymsz/SZpacklib/; make tidy
