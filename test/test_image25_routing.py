from __future__ import annotations

import unittest
from unittest import mock

from services import openai_backend_api
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI
from services.protocol import openai_v1_models
from utils.helper import is_image_chat_request, is_supported_image_model, split_image_model


class Image25RoutingTests(unittest.TestCase):
    def setUp(self):
        self.api = OpenAIBackendAPI("")
        self.addCleanup(self.api.close)

    def test_explicit_slugs_are_supported(self):
        for name in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
            self.assertTrue(is_supported_image_model(name))
            self.assertEqual(split_image_model(name), (None, name))

    def test_bare_alias_is_normalized_to_flare(self):
        self.assertEqual(split_image_model(" GPT-IMAGE-2.5 "), (None, "gpt-image-2.5-flare"))
        self.assertTrue(is_supported_image_model("gpt-image-2.5"))

    def test_unknown_slug_is_not_accepted(self):
        self.assertFalse(is_supported_image_model("gpt-image-2.5-unknown"))
        self.assertFalse(is_supported_image_model("plus-gpt-image-2.5-flare"))

    def test_image25_chat_requests_use_the_image_route(self):
        for name in ("gpt-image-2.5", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
            self.assertTrue(is_image_chat_request({"model": name}))

    def test_default_web_image_model_uses_auto(self):
        """合并说明：本测试原属 #420，断言 gpt-image-2 沿用
        default_upstream_model_name（("gpt-5-5", "extended")）。

        但 #418 基于 2026-10 网页端抓包，明确要求默认网页生图用
        `model=auto` 由服务端挑模型，且 prepare 与 f/conversation 必须共用
        这一取值。两者直接冲突，合并时以 #418 为准（它才是修复空 404 的完整
        改动，#420 不含 sentinel 相关修改、单独无法恢复生图）。

        显式的 Image 2.5 slug 仍然会原样透传，见
        test_prepare_generate_and_edit_preserve_the_same_model。
        """
        with mock.patch.object(openai_backend_api, "config", mock.Mock(
                default_upstream_model_name="gpt-5-5-extended", default_thinking_effort="auto")):
            self.assertEqual(self.api._image_model_settings("gpt-image-2"), ("auto", ""))

    def test_codex_prefixes_keep_their_existing_route(self):
        for prefix in ("plus", "team", "pro"):
            name = f"{prefix}-codex-gpt-image-2"
            self.assertEqual(split_image_model(name), (prefix, "codex-gpt-image-2"))
            self.assertEqual(self.api._image_model_settings(name)[0], "codex-gpt-image-2")

    def test_prepare_generate_and_edit_preserve_the_same_model(self):
        """合并说明：默认（未配置 default_upstream_model_name_25）时，网页生图
        一律走 `model=auto`（#418 的 2026-10 抓包口径）；只有显式配置了该名字，
        2.5 slug 才会按配置上送。这里测的是后者（配置逃生口）。

        默认走 auto 的行为由 test_sentinel_handshake.ImageModelSlugTests
        .test_web_image_models_map_to_auto 覆盖。
        """
        requirements = ChatRequirements(token="requirements", turnstile_token="turnstile")
        response = mock.Mock(status_code=200)
        response.json.return_value = {"conduit_token": "conduit"}
        reference = {"file_id": "file-1", "width": 10, "height": 10,
                     "file_size": 20, "mime_type": "image/png", "file_name": "input.png"}
        for requested, configured in (
            ("gpt-image-2.5", "gpt-image-2.5-flare"),
            ("gpt-image-2.5-flare", "gpt-image-2.5-flare"),
            ("gpt-image-2.5-sunburst", "gpt-image-2.5-sunburst"),
        ):
            for references in (None, [reference]):
                with self.subTest(requested=requested, edit=bool(references)):
                    with mock.patch.object(openai_backend_api, "config", mock.Mock(
                            default_upstream_model_name_25=configured,
                            default_thinking_effort="auto")), \
                            mock.patch.object(self.api.session, "post", return_value=response) as post:
                        self.api._prepare_image_conversation("apple", requirements, requested)
                        self.api._start_image_generation("apple", requirements, "conduit", requested, references)
                    first, second = post.call_args_list
                    self.assertEqual(first.kwargs["json"]["model"], configured)
                    self.assertEqual(second.kwargs["json"]["model"], configured)

    def test_catalog_advertises_image25_when_accounts_exist(self):
        with mock.patch.object(openai_v1_models.model_catalog_service, "list_models",
                               return_value={"data": []}), mock.patch.object(
                openai_v1_models.account_service, "list_accounts", return_value=[{}]):
            names = {item["id"] for item in openai_v1_models.list_models()["data"]}
        self.assertTrue({"gpt-image-2.5", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"} <= names)

    def test_catalog_does_not_advertise_image25_without_accounts(self):
        with mock.patch.object(openai_v1_models.model_catalog_service, "list_models",
                               return_value={"data": []}), mock.patch.object(
                openai_v1_models.account_service, "list_accounts", return_value=[]):
            self.assertEqual(openai_v1_models.list_models()["data"], [])

    def test_existing_catalog_entries_are_not_duplicated_or_replaced(self):
        original = {"id": "gpt-image-2.5-flare", "owned_by": "upstream"}
        with mock.patch.object(openai_v1_models.model_catalog_service, "list_models",
                               return_value={"data": [original]}), mock.patch.object(
                openai_v1_models.account_service, "list_accounts", return_value=[{}]):
            data = openai_v1_models.list_models()["data"]
        matches = [item for item in data if item["id"] == original["id"]]
        self.assertEqual(matches, [original])


if __name__ == "__main__":
    unittest.main()
