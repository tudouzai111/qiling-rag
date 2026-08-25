# file_uploader.py MinIO Python SDK example
from minio import Minio
from minio.error import S3Error


def main():
    # 实例化MinIO客户端
    client = Minio("127.0.0.1:9000",
                   access_key="minioadmin",
                   secret_key="minioadmin",
                   secure=False
                   )

    # 上传的文件地址
    source_file = r"C:\Users\TuDouZai\Desktop\大狗叫.jpeg"

    # 桶名
    bucket_name = "python-test-bucket"

    # 对象名
    destination_file = "my-test-img.jpeg"

    # 判断桶是否存在
    found = client.bucket_exists(bucket_name)
    if not found:
        # 创建桶
        client.make_bucket(bucket_name)
        print("创建桶：", bucket_name)
    else:
        print("桶：", bucket_name, "已存在！")

    # 上传文件
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        source_file, "上传成功！",
        destination_file, "上传到桶：", bucket_name,
    )


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)
