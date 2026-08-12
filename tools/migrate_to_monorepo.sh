#!/bin/bash
# ==============================================================================
# Monorepo Migration Script
# Converts git submodules into directly-tracked monorepo code
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== Phase 1: Remove CMake-only submodules ==="

# Remove catkin (CMake build system - not needed with Bazel)
if [ -d "lib/catkin" ]; then
    echo "  Removing lib/catkin..."
    git rm -rf lib/catkin 2>/dev/null || rm -rf lib/catkin
fi

# Remove cmake_modules (CMake modules - not needed with Bazel)
if [ -d "lib/cmake_modules" ]; then
    echo "  Removing lib/cmake_modules..."
    git rm -rf lib/cmake_modules 2>/dev/null || rm -rf lib/cmake_modules
fi

echo "=== Phase 2: Remove magic_enum submodule (replaced by bzlmod) ==="
if [ -d "lib/magic_enum" ]; then
    echo "  Removing lib/magic_enum..."
    git rm -rf lib/magic_enum 2>/dev/null || rm -rf lib/magic_enum
fi

echo "=== Phase 3: Initialize and absorb mujoco submodule ==="
if grep -q "lib/mujoco" .gitmodules 2>/dev/null; then
    echo "  Initializing lib/mujoco submodule..."
    git submodule update --init lib/mujoco 2>/dev/null || true

    if [ -d "lib/mujoco" ]; then
        echo "  Absorbing lib/mujoco into monorepo..."
        # Save the code
        cp -r lib/mujoco /tmp/mujoco_backup
        rm -rf /tmp/mujoco_backup/.git

        # Remove as submodule
        git rm -rf lib/mujoco 2>/dev/null || true
        rm -rf .git/modules/lib/mujoco 2>/dev/null || true

        # Re-add as regular directory
        mv /tmp/mujoco_backup lib/mujoco
        git add lib/mujoco
        echo "  Done — lib/mujoco is now a regular directory"
    else
        echo "  Warning: lib/mujoco could not be initialized, skipping"
    fi
fi

echo "=== Phase 4: Absorb ocs2_ros2 and flatten to lib/ocs2/ ==="
if [ -d "lib/ocs2_ros2" ]; then
    echo "  Saving ocs2_ros2 code..."
    cp -r lib/ocs2_ros2 /tmp/ocs2_ros2_backup
    rm -rf /tmp/ocs2_ros2_backup/.git

    echo "  Removing ocs2_ros2 submodule..."
    git rm -rf lib/ocs2_ros2 2>/dev/null || rm -rf lib/ocs2_ros2
    rm -rf .git/modules/lib/ocs2_ros2 2>/dev/null || true

    echo "  Creating flattened lib/ocs2/ directory..."
    mkdir -p lib/ocs2

    # Move and rename sub-packages (strip ocs2_ prefix)
    for dir in /tmp/ocs2_ros2_backup/ocs2_*/; do
        dirname=$(basename "$dir")
        # Strip ocs2_ prefix
        newname="${dirname#ocs2_}"
        echo "    $dirname → $newname"
        mv "$dir" "lib/ocs2/$newname"
    done

    # Move ocs2_pinocchio sub-packages
    if [ -d "lib/ocs2/pinocchio" ]; then
        echo "  Flattening pinocchio sub-packages..."
        for dir in lib/ocs2/pinocchio/ocs2_*/; do
            if [ -d "$dir" ]; then
                dirname=$(basename "$dir")
                newname="${dirname#ocs2_}"
                echo "    pinocchio/$dirname → pinocchio/$newname"
                mv "$dir" "lib/ocs2/pinocchio/$newname"
            fi
        done
    fi

    # Move ocs2_sqp sub-packages
    if [ -d "lib/ocs2/sqp/ocs2_sqp" ]; then
        echo "  Flattening sqp sub-packages..."
        # ocs2_sqp/ocs2_sqp → sqp/sqp (the actual SQP solver code)
        mv "lib/ocs2/sqp/ocs2_sqp" "lib/ocs2/sqp/sqp"
    fi

    # Move ocs2_test_tools sub-packages
    if [ -d "lib/ocs2/test_tools" ]; then
        echo "  Flattening test_tools sub-packages..."
        for dir in lib/ocs2/test_tools/ocs2_*/; do
            if [ -d "$dir" ]; then
                dirname=$(basename "$dir")
                newname="${dirname#ocs2_}"
                echo "    test_tools/$dirname → test_tools/$newname"
                mv "$dir" "lib/ocs2/test_tools/$newname"
            fi
        done
    fi

    # Move ocs2_robotic_examples sub-packages
    if [ -d "lib/ocs2/robotic_examples" ]; then
        echo "  Flattening robotic_examples sub-packages..."
        for dir in lib/ocs2/robotic_examples/ocs2_*/; do
            if [ -d "$dir" ]; then
                dirname=$(basename "$dir")
                newname="${dirname#ocs2_}"
                echo "    robotic_examples/$dirname → robotic_examples/$newname"
                mv "$dir" "lib/ocs2/robotic_examples/$newname"
            fi
        done
    fi

    # Move ocs2_ros_interfaces (ROS1 legacy)
    # Keep as-is since it doesn't follow the ocs2_ pattern consistently

    # Copy BUILD.bazel (will be updated later)
    if [ -f "/tmp/ocs2_ros2_backup/BUILD.bazel" ]; then
        cp "/tmp/ocs2_ros2_backup/BUILD.bazel" "lib/ocs2/BUILD.bazel"
    fi

    # Copy any other root-level files
    for f in /tmp/ocs2_ros2_backup/LICENSE* /tmp/ocs2_ros2_backup/README*; do
        [ -f "$f" ] && cp "$f" lib/ocs2/
    done

    git add lib/ocs2
    echo "  Done — lib/ocs2 is now a flattened monorepo directory"

    # Cleanup
    rm -rf /tmp/ocs2_ros2_backup
fi

echo "=== Phase 5: Delete all CMakeLists.txt and package.xml files ==="
find . -name "CMakeLists.txt" -not -path "./.git/*" -delete -print 2>/dev/null | sed 's/^/  Deleted: /'
find . -name "package.xml" -not -path "./.git/*" -delete -print 2>/dev/null | sed 's/^/  Deleted: /'

echo "=== Phase 6: Clean up .gitmodules ==="
if [ -f ".gitmodules" ]; then
    rm .gitmodules
    git rm --cached .gitmodules 2>/dev/null || true
    echo "  Deleted .gitmodules"
fi

echo "=== Phase 7: Clean up colcon artifacts ==="
rm -f colcon.defaults.yaml build_all.log
[ -f "Makefile.bazel" ] && mv Makefile.bazel Makefile && echo "  Renamed Makefile.bazel → Makefile"

echo "=== Phase 8: Stage all changes ==="
git add -A

echo ""
echo "=== Migration complete! ==="
echo "Review with 'git status' and 'git diff --cached --stat', then commit."
echo ""
echo "Next steps:"
echo "  1. Update lib/ocs2/BUILD.bazel with new flattened paths"
echo "  2. Update humanoid_nmpc BUILD files: //lib/ocs2_ros2: → //lib/ocs2:"
echo "  3. Run 'bazel build //...' to verify"
