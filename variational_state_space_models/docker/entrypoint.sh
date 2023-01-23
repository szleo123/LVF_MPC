#!/usr/bin/env bash
groupadd --gid $EXT_GID $EXT_USER
useradd -d /home/user -s /usr/bin/fish --uid $EXT_UID -g $EXT_USER $EXT_USER
cd /home/user/code
su $EXT_USER
