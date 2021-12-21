export http_proxy=http://proxy-outer.dl.fuxi.netease.com:8787
export https_proxy=http://proxy-outer.dl.fuxi.netease.com:8787

python3.6 -m pip install /root/to_install_packages/zh_core_web_trf-3.2.0.tar.gz
python3.6 -m pip install /root/to_install_packages/en_core_web_trf-3.2.0.tar.gz
python3.6 -m pip install /root/to_install_packages/zh_core_web_sm-3.2.0.tar.gz
python3.6 -m pip install /root/to_install_packages/en_core_web_sm-3.2.0.tar.gz
python3.6 -m pip install transformers==4.11.3
python3.6 -m pip install benepar==0.2.0