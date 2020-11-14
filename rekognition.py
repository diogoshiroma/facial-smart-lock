import boto3

# if __name__ == "__main__":

    # bucket='facialsmartlock2'
    # collectionId='smartlockcollection'
    # filename='rogerio-oculos.jpg'
    # threshold = 70
    # maxFaces=2

    # client=boto3.client('rekognition')
    # image=open(filename, 'rb')

    # response=client.search_faces_by_image(CollectionId=collectionId,
    #                             # Image={'S3Object':{'Bucket':bucket,'Name':fileName}},
    #                             Image={'Bytes': image.read()},
    #                             FaceMatchThreshold=threshold,
    #                             MaxFaces=maxFaces)

                                
    # faceMatches=response['FaceMatches']
    # print ('Matching faces')
    # if faceMatches:
    #     for match in faceMatches:
    #             # print ('FaceId:' + match['Face']['FaceId'])
    #             print ('FaceId:' + match['Face']['ExternalImageId'])
    #             print ('Similarity: ' + "{:.2f}".format(match['Similarity']) + "%")
    # else:
    #     print('No match!')

def aws_rekognition():
    bucket='facialsmartlock2'
    collectionId='smartlockcollection'
    filename='potato.jpg'
    threshold = 70
    maxFaces=2

    client=boto3.client('rekognition')
    image=open(filename, 'rb')

    response=client.search_faces_by_image(CollectionId=collectionId,
                                # Image={'S3Object':{'Bucket':bucket,'Name':fileName}},
                                Image={'Bytes': image.read()},
                                FaceMatchThreshold=threshold,
                                MaxFaces=maxFaces)

                                
    faceMatches=response['FaceMatches']
    print ('Matching faces')
    if faceMatches:
        for match in faceMatches:
                # print ('FaceId:' + match['Face']['FaceId'])
                print ('FaceId:' + match['Face']['ExternalImageId'])
                print ('Similarity: ' + "{:.2f}".format(match['Similarity']) + "%")
    else:
        print('No match!')
