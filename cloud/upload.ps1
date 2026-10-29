param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$SshTarget,
    [ValidateRange(1,65535)][int]$SshPort = 22,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\hiwonder_cloud_ed25519_20261029",
    [string]$RemoteDirectory = '/workspace/hiwonder-upload'
)
$ErrorActionPreference = 'Stop'
if ($SshTarget -notmatch '^(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9][A-Za-z0-9.-]*$') {
    throw 'Expected an SSH user@IPv4-or-hostname target.'
}
if ($RemoteDirectory -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteDirectory.Split('/') -contains '..' -or $RemoteDirectory -eq '/') {
    throw 'Use a simple absolute project upload directory.'
}
$archivePath = (Resolve-Path -LiteralPath $Bundle).Path
$keyPath = (Resolve-Path -LiteralPath $IdentityFile).Path
$archiveName = [IO.Path]::GetFileName($archivePath)
if ($archiveName -notmatch '^[A-Za-z0-9._-]+\.tar\.gz$') { throw 'Use a simple .tar.gz filename.' }
$expectedHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$sshOptions = @('-i', $keyPath, '-p', "$SshPort", '-o', 'ForwardAgent=no', '-o', 'StrictHostKeyChecking=ask')
& ssh.exe @sshOptions $SshTarget "mkdir -p -- '$RemoteDirectory'"
if ($LASTEXITCODE -ne 0) { throw 'SSH directory preparation failed.' }
$scpTarget = $SshTarget + ':' + $RemoteDirectory + '/' + $archiveName
& scp.exe -i $keyPath -P $SshPort -o ForwardAgent=no -o StrictHostKeyChecking=ask $archivePath $scpTarget
if ($LASTEXITCODE -ne 0) { throw 'Archive upload failed.' }
$remoteHash = & ssh.exe @sshOptions $SshTarget "sha256sum -- '$RemoteDirectory/$archiveName'"
if ($LASTEXITCODE -ne 0) { throw 'Remote checksum failed.' }
$match = $remoteHash | Select-String -Pattern '^([0-9a-fA-F]{64})\s' | Select-Object -First 1
if (-not $match -or $match.Matches[0].Groups[1].Value.ToLowerInvariant() -ne $expectedHash) { throw 'Uploaded archive checksum differs.' }
Write-Output "Verified upload: $RemoteDirectory/$archiveName"
Write-Output "SHA256: $expectedHash"
Write-Output 'Only files were uploaded; no training, purchase or GPU firmware change was started.'
