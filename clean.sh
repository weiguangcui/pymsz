git clean -fx
rm -rf build
rm -rf pymsz.egg-info
rm -rf pymsz/*.pyc
cd pymsz/SZpacklib/; make tidy
