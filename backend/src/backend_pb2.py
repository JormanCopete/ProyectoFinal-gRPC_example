# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: backend.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rbackend.proto\";\n\x05image\x12\x13\n\x0bimg_content\x18\x01 \x01(\t\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\"\x18\n\x08img_path\x12\x0c\n\x04path\x18\x01 \x01(\t\"-\n\x11prediction_result\x12\x18\n\x10label_prediction\x18\x01 \x01(\t\"/\n\x0fimg_predic_ruta\x12\x1c\n\x14path_img_predic_ruta\x18\x01 \x01(\t2[\n\x07\x42\x61\x63kend\x12\x1f\n\nload_image\x12\t.img_path\x1a\x06.image\x12/\n\x07predict\x12\x10.img_predic_ruta\x1a\x12.prediction_resultb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'backend_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _IMAGE._serialized_start=17
  _IMAGE._serialized_end=76
  _IMG_PATH._serialized_start=78
  _IMG_PATH._serialized_end=102
  _PREDICTION_RESULT._serialized_start=104
  _PREDICTION_RESULT._serialized_end=149
  _IMG_PREDIC_RUTA._serialized_start=151
  _IMG_PREDIC_RUTA._serialized_end=198
  _BACKEND._serialized_start=200
  _BACKEND._serialized_end=291
# @@protoc_insertion_point(module_scope)