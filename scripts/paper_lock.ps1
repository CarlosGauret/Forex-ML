function Read-LockPid($path) {
    try {
        return (Get-Content -LiteralPath $path -Raw -ErrorAction Stop).Trim()
    } catch {
        return ""
    }
}

function Get-ProcessInfoByPid($processId) {
    $procInfo = $null
    try {
        $procInfo = Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f [int]$processId) -ErrorAction Stop
    } catch {
    }
    if ($null -ne $procInfo) {
        return $procInfo
    }

    try {
        $proc = Get-Process -Id ([int]$processId) -ErrorAction Stop
        return [pscustomobject]@{
            Name = ($proc.ProcessName + ".exe")
            CommandLine = $null
            ExecutablePath = $proc.Path
            ParentProcessId = $null
        }
    } catch {
        return $null
    }
}

function Test-LockFileHeld($path) {
    $probe = $null
    try {
        $probe = [System.IO.File]::Open(
            $path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
        $probe.Close()
        return $false
    } catch {
        if ($null -ne $probe) {
            try { $probe.Close() } catch {}
        }
        return $true
    }
}

function Test-ForexMlRunnerProcess($processId) {
    $procInfo = Get-ProcessInfoByPid $processId
    if ($null -eq $procInfo) {
        return $false
    }

    $name = ([string]$procInfo.Name).ToLowerInvariant()
    $allowedNames = @("powershell.exe", "pwsh.exe", "cmd.exe", "python.exe", "pythonw.exe")
    if ($allowedNames -notcontains $name) {
        return $false
    }

    $projectNorm = [System.IO.Path]::GetFullPath($project).TrimEnd("\").ToLowerInvariant()
    $texts = New-Object System.Collections.Generic.List[string]
    if ($procInfo.CommandLine) { $texts.Add([string]$procInfo.CommandLine) }
    if ($procInfo.ExecutablePath) { $texts.Add([string]$procInfo.ExecutablePath) }

    $parentInfo = Get-ProcessInfoByPid $procInfo.ParentProcessId
    if ($null -ne $parentInfo) {
        if ($parentInfo.CommandLine) { $texts.Add([string]$parentInfo.CommandLine) }
        if ($parentInfo.ExecutablePath) { $texts.Add([string]$parentInfo.ExecutablePath) }
    }

    foreach ($text in $texts) {
        $lower = $text.ToLowerInvariant()
        if ($lower.Contains("run_paper.bat")) { return $true }
        if ($lower.Contains($projectNorm)) { return $true }
        if ($lower.Contains("forex ml")) { return $true }
    }

    return $false
}

function New-PaperLock($path, $processId) {
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        if (Test-Path -LiteralPath $path) {
            $lockPid = Read-LockPid $path
            $activeRunner = $false
            $processExists = $false

            if ($lockPid -match "^\d+$") {
                $processExists = $null -ne (Get-ProcessInfoByPid ([int]$lockPid))
                if ($processExists) {
                    $activeRunner = Test-ForexMlRunnerProcess ([int]$lockPid)
                }
            }

            if ((-not $activeRunner) -and $processExists -and (Test-LockFileHeld $path)) {
                $activeRunner = $true
            }

            if ($activeRunner) {
                Write-Log $summaryLog ("PAPER ya esta en ejecucion. PID activo: " + $lockPid)
                Write-Log $summaryLog "RUNNER EXIT CODE: 0"
                Write-Log $summaryLog ""
                return $null
            }

            Write-Log $summaryLog "STALE LOCK DETECTED"
            Write-Log $summaryLog ("LOCK PID: " + $lockPid)
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 50
            continue
        }

        $stream = $null
        try {
            $stream = [System.IO.File]::Open(
                $path,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::Read
            )
            $bytes = [System.Text.Encoding]::ASCII.GetBytes([string]$processId)
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush()
            return $stream
        } catch {
            if ($null -ne $stream) {
                try { $stream.Close() } catch {}
            }
            Start-Sleep -Milliseconds 100
        }
    }

    throw "No se pudo crear paper lock de forma atomica"
}

function Clear-OwnPaperLock($path, $processId, $stream) {
    if ($null -ne $stream) {
        try { $stream.Close() } catch {}
    }

    if (Test-Path -LiteralPath $path) {
        $savedPid = Read-LockPid $path
        if ($savedPid -eq ([string]$processId)) {
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }
}
