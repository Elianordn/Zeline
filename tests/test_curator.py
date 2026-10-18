"""SkillForge Curator: pin/unpin, archive/restore, usage summary, prune.

These tests cover the curator lifecycle without touching the agent loop,
gateway, or tool dispatch — purely ForgeStore + skills data operations.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class CuratorTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name).resolve()
        self.env = mock.patch.dict(os.environ, {
            "ZELINE_HOME": str(self.home / "state"),
            "ZELINE_API_KEY": "fixture-only",
            "ZELINE_MODEL": "fixture",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        # Force reimport with clean state
        for name in list(sys.modules):
            if name == "zeline" or name.startswith("zeline."):
                del sys.modules[name]
        self.skillforge = __import__("zeline.skillforge", fromlist=["ForgeStore"])
        self.skills = __import__("zeline.skills", fromlist=["manage_skill"])
        self.skills._ensure_dirs()
        self.store = self.skillforge.ForgeStore()

    def _create_private_skill(self, name, content="# Test\n> test skill"):
        result = self.skills.manage_skill("create", name=name, content=content)
        self.assertIn("OK", result)
        return result


class PinTests(CuratorTestBase):
    def test_pin_and_unpin_skill(self):
        self._create_private_skill("alpha")
        result = self.store.pin("alpha")
        self.assertIn("Pinned", result)
        self.assertTrue(self.store.is_pinned("alpha"))
        self.assertIn("alpha", self.store.pinned())

        result = self.store.unpin("alpha")
        self.assertIn("Unpinned", result)
        self.assertFalse(self.store.is_pinned("alpha"))
        self.assertNotIn("alpha", self.store.pinned())

    def test_pin_nonexistent_skill_raises(self):
        with self.assertRaises(ValueError):
            self.store.pin("nonexistent")

    def test_unpin_nonexistent_is_safe(self):
        # Unpin a skill that was never pinned — should not raise
        result = self.store.unpin("never-pinned")
        self.assertIn("Unpinned", result)

    def test_pinned_skill_refuses_delete(self):
        self._create_private_skill("beta")
        self.store.pin("beta")
        result = self.skills.manage_skill("delete", name="beta")
        self.assertIn("pinned", result.lower())
        # Skill still exists
        self.assertIsNotNone(self.skills._locate_unit("beta"))

    def test_unpinned_skill_allows_delete(self):
        self._create_private_skill("gamma")
        self.store.pin("gamma")
        self.store.unpin("gamma")
        result = self.skills.manage_skill("delete", name="gamma")
        self.assertIn("Deleted", result)

    def test_pins_persist_across_store_instances(self):
        self._create_private_skill("delta")
        self.store.pin("delta")
        store2 = self.skillforge.ForgeStore()
        self.assertTrue(store2.is_pinned("delta"))


class ArchiveTests(CuratorTestBase):
    def test_archive_and_restore(self):
        self._create_private_skill("archivable")
        result = self.store.archive("archivable")
        self.assertIn("Archived", result)
        self.assertIn("archivable", self.store.archived())
        # Skill no longer active
        self.assertIsNone(self.skills._locate_unit("archivable"))

        result = self.store.restore("archivable")
        self.assertIn("Restored", result)
        self.assertNotIn("archivable", self.store.archived())
        # Skill is active again
        self.assertIsNotNone(self.skills._locate_unit("archivable"))

    def test_archive_pinned_skill_raises(self):
        self._create_private_skill("pinned-one")
        self.store.pin("pinned-one")
        with self.assertRaises(ValueError) as ctx:
            self.store.archive("pinned-one")
        self.assertIn("pinned", str(ctx.exception).lower())

    def test_archive_public_skill_raises(self):
        # Public skills cannot be archived
        with self.assertRaises(ValueError) as ctx:
            self.store.archive("zeline-zenith-z0")
        # Either "not found" or "bundled/public"
        self.assertTrue("not found" in str(ctx.exception).lower()
                        or "public" in str(ctx.exception).lower())

    def test_archive_nonexistent_raises(self):
        with self.assertRaises(ValueError):
            self.store.archive("does-not-exist")

    def test_restore_nonexistent_raises(self):
        with self.assertRaises(ValueError):
            self.store.restore("never-archived")

    def test_double_archive_raises(self):
        self._create_private_skill("double")
        self.store.archive("double")
        with self.assertRaises(ValueError) as ctx:
            self.store.archive("double")
        # Second attempt fails — skill is no longer in active scope
        msg = str(ctx.exception).lower()
        self.assertTrue("not found" in msg or "already archived" in msg)

    def test_restore_when_active_raises(self):
        self._create_private_skill("conflict")
        # Archive it
        self.store.archive("conflict")
        # Recreate with same name
        self._create_private_skill("conflict")
        with self.assertRaises(ValueError) as ctx:
            self.store.restore("conflict")
        self.assertIn("already exists", str(ctx.exception).lower())


class UsageSummaryTests(CuratorTestBase):
    def test_empty_usage(self):
        result = self.store.usage_summary()
        self.assertEqual(result, [])

    def test_usage_with_records(self):
        now = time.time()
        rev = "a" * 64
        self.store.record_usage("u1", "t1", "alpha", "private", rev, "success", now=now)
        self.store.record_usage("u1", "t2", "alpha", "private", rev, "failure", now=now - 100)
        self.store.record_usage("u1", "t3", "alpha", "private", rev, "success", now=now - 200)

        result = self.store.usage_summary("alpha", now=now)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "alpha")
        self.assertEqual(result[0]["total"], 3)
        self.assertEqual(result[0]["success"], 2)
        self.assertEqual(result[0]["failure"], 1)
        self.assertEqual(result[0]["success_rate"], 66.7)

    def test_usage_all_skills(self):
        now = time.time()
        rev = "b" * 64
        self.store.record_usage("u1", "t1", "alpha", "private", rev, "success", now=now)
        self.store.record_usage("u1", "t2", "beta", "private", rev, "failure", now=now)

        result = self.store.usage_summary(now=now)
        self.assertEqual(len(result), 2)
        names = {r["name"] for r in result}
        self.assertEqual(names, {"alpha", "beta"})

    def test_usage_respects_days_window(self):
        now = time.time()
        rev = "c" * 64
        # Old record outside window
        self.store.record_usage("u1", "t1", "old-skill", "private", rev, "success",
                                now=now - 100 * 86400)
        result = self.store.usage_summary("old-skill", days=90, now=now)
        self.assertEqual(result, [])


class PruneTests(CuratorTestBase):
    def test_prune_dry_run(self):
        self._create_private_skill("stale-skill")
        # Backdate mtime to simulate staleness
        skill_path = self.skills.PRIVATE_SKILLS_DIR / "stale-skill" / "SKILL.md"
        old_time = time.time() - 200 * 86400
        os.utime(skill_path, (old_time, old_time))

        result = self.store.prune(stale_days=180, dry_run=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "stale-skill")
        self.assertEqual(result[0]["action"], "would_archive")
        # Skill still exists (dry run)
        self.assertIsNotNone(self.skills._locate_unit("stale-skill"))

    def test_prune_execute(self):
        self._create_private_skill("dead-skill")
        skill_path = self.skills.PRIVATE_SKILLS_DIR / "dead-skill" / "SKILL.md"
        old_time = time.time() - 200 * 86400
        os.utime(skill_path, (old_time, old_time))

        result = self.store.prune(stale_days=180, dry_run=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["action"], "archived")
        # Skill moved to archive
        self.assertIsNone(self.skills._locate_unit("dead-skill"))
        self.assertIn("dead-skill", self.store.archived())

    def test_prune_skips_pinned(self):
        self._create_private_skill("pinned-old")
        self.store.pin("pinned-old")
        skill_path = self.skills.PRIVATE_SKILLS_DIR / "pinned-old" / "SKILL.md"
        old_time = time.time() - 200 * 86400
        os.utime(skill_path, (old_time, old_time))

        result = self.store.prune(stale_days=180, dry_run=False)
        self.assertEqual(result, [])
        # Skill still exists
        self.assertIsNotNone(self.skills._locate_unit("pinned-old"))

    def test_prune_skips_recently_used(self):
        self._create_private_skill("used-old")
        skill_path = self.skills.PRIVATE_SKILLS_DIR / "used-old" / "SKILL.md"
        now = time.time()
        old_time = now - 200 * 86400
        os.utime(skill_path, (old_time, old_time))
        # Record recent usage
        rev = "d" * 64
        self.store.record_usage("u1", "t1", "used-old", "private", rev, "success", now=now - 10)

        result = self.store.prune(stale_days=180, dry_run=True, now=now)
        self.assertEqual(result, [])

    def test_prune_skips_recent_skills(self):
        self._create_private_skill("fresh-skill")
        # mtime is now, so it's not stale
        result = self.store.prune(stale_days=180, dry_run=True)
        self.assertEqual(result, [])


class CLITests(CuratorTestBase):
    def _run(self, *args):
        import argparse
        parser = argparse.ArgumentParser()
        self.skillforge.add_arguments(parser)
        parsed = parser.parse_args(list(args))
        return self.skillforge.command(parsed)

    def test_pin_unpin_pinned_cli(self):
        self._create_private_skill("cli-pin")
        self.assertEqual(self._run("pin", "cli-pin"), 0)
        self.assertEqual(self._run("pinned"), 0)
        self.assertEqual(self._run("unpin", "cli-pin"), 0)

    def test_archive_restore_archived_cli(self):
        self._create_private_skill("cli-arch")
        self.assertEqual(self._run("archive", "cli-arch"), 0)
        self.assertEqual(self._run("archived"), 0)
        self.assertEqual(self._run("restore", "cli-arch"), 0)

    def test_usage_cli(self):
        self.assertEqual(self._run("usage"), 0)
        self.assertEqual(self._run("usage", "--days", "30"), 0)

    def test_prune_dry_run_cli(self):
        self.assertEqual(self._run("prune"), 0)


if __name__ == "__main__":
    unittest.main()
