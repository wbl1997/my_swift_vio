The ORBvoc.yaml could be downloaded from [dropbox](http://www.dropbox.com/s/lyo0qgbdxn6eg6o/ORBvoc.zip?dl=1)
which is not accessible without tunneling.
So I have shared its copy on [onedrive](https://1drv.ms/u/s!Atcp9ufHvhWXa_XKz3127AoVPuA?e=3OkNZR).

ORBvoc.yml was created by the authors of ORBSLAM2, and it has the same size as
the yml vocabulary used in an earlier commit of [ORB-SLAM](https://github.com/raulmur/ORB_SLAM).
though their md5sum are different.

The latest version of ORB-SLAM and ORB-SLAM2 both used the same
[ORBvoc.txt](https://github.com/raulmur/ORB_SLAM2/blob/master/Vocabulary/ORBvoc.txt.tar.gz)
which is smaller than the ORBvoc.yml.
By visual inspection, all these files have exactly the same content.
Unfortunately, in mainland China, wget/cmake download fails to download ORBvoc.txt from github,
or ORBvoc.yml from onedrive.
