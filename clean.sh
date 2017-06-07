git clean -fx
rm -rf build
rm -rf pymsz.egg-info
rm -rf pymsz/*.pyc
cd pymsz/SZpack.v1.1.1/; make tidy
