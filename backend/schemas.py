from typing import Literal
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
import re

# ---------------------------------------------------------------------------
# Disposable / temporary email provider blocklist (400+ domains).
# Sources: disposable-email-domains, burner-email-providers, stopforumspam.
# ---------------------------------------------------------------------------
_DISPOSABLE_DOMAINS: frozenset[str] = frozenset({
    # --- classic disposable ---
    "mailinator.com", "guerrillamail.com", "guerrillamail.info",
    "guerrillamail.biz", "guerrillamail.de", "guerrillamail.net",
    "guerrillamail.org", "grr.la", "spam4.me", "trashmail.com",
    "trashmail.me", "trashmail.net", "trashmail.at", "trashmail.io",
    "trashmail.org", "trashmail.xyz", "yopmail.com", "yopmail.fr",
    "cool.fr.nf", "jetable.fr.nf", "nospam.ze.tc", "nomail.xl.cx",
    "mega.zik.dj", "speed.1s.fr", "courriel.fr.nf", "moncourrier.fr.nf",
    "monemail.fr.nf", "monmail.fr.nf", "jetable.com",
    "10minutemail.com", "10minutemail.net", "10minutemail.org",
    "10minutemail.co.za", "10minutemail.de", "10minemail.com",
    "10minutmail.com", "20minutemail.com", "30minutemail.com",
    "60minutemail.com", "fakeinbox.com", "mailnull.com", "spamgourmet.com",
    "spamgourmet.net", "spamgourmet.org", "spamgourmet.me",
    "tempmail.com", "temp-mail.org", "temp-mail.ru", "tempinbox.com",
    "dispostable.com", "discard.email", "throwam.com", "throwam.net",
    "throwaway.email", "maildrop.cc", "mailnesia.com", "mailnesia.me",
    "sharklasers.com", "guerrillamailblock.com", "spam.la",
    "cuvox.de", "dayrep.com", "einrot.com", "fleckens.hu",
    "gustr.com", "jourrapide.com", "krovatk.ru", "objectmail.com",
    "obobbo.com", "rhyta.com", "superrito.com", "teleworm.us",
    "armyspy.com", "cuvox.de", "dayrep.com", "einrot.com",
    "fleckens.hu", "gustr.com", "jourrapide.com", "rhyta.com",
    "superrito.com", "teleworm.us", "armyspy.com",
    "spamhereplease.com", "spamhereplease.net", "nobulk.com",
    "mailnew.com", "spamfree24.org", "spamfree.eu", "kasmail.com",
    "spamless.nl", "wegwerfmail.de", "wegwerfmail.net", "wegwerfmail.org",
    "byom.de", "emailaaa.com", "emailias.com", "emailinfive.com",
    "emailmiser.com", "emailsensei.com", "emailtemporanea.com",
    "emailtemporanea.net", "emailtemporario.com.br",
    "emailto.de", "emailwarden.com", "emailx.at.hm", "emailxfer.com",
    "emailz.cf", "emailz.ga", "emailz.ml", "etranquil.com",
    "etranquil.net", "etranquil.org", "explodemail.com", "eyepaste.com",
    "fakeinbox.cf", "fakeinbox.com", "fakeinbox.ga", "fakeinbox.info",
    "fakeinbox.ml", "fakeinbox.tk", "fakemailgenerator.com",
    "fakethat.com", "filzmail.com", "filzmail.de", "fivemail.de",
    "fleckens.hu", "frapmail.com", "freundin.ru", "fuckingduh.com",
    "fudgerub.com", "fuirio.com", "gamail.top", "garliclife.com",
    "gelitik.in", "get1mail.com", "getairmail.com", "getairmail.cf",
    "getairmail.ga", "getairmail.gq", "getairmail.ml", "getairmail.tk",
    "getfun.men", "getmails.eu", "getonemail.com", "getonemail.net",
    "ghosttexter.de", "giantmail.de", "girlsundertheinfluence.com",
    "gishpuppy.com", "gmailni.com", "gms.pl", "gorillaswithdirtyarmpits.com",
    "gotmail.com", "gotmail.net", "gotmail.org", "gowikibooks.com",
    "gowikicampus.com", "gowikicars.com", "gowikifilms.com",
    "gowikigames.com", "gowikimusic.com", "gowikinetwork.com",
    "gowikitravel.com", "gowikitv.com", "grandmamail.com",
    "great-host.in", "greensloth.com", "grr.la", "gsrv.co.uk",
    "guessmail.com", "gun.io", "gustr.com",
    "h8s.org", "hailmail.net", "harakirimail.com", "hat-geld.de",
    "hatespam.org", "herp.in", "hidemail.de", "hmamail.com",
    "hochsitze.com", "hopemail.biz", "hot-mail.co", "hotmai1.com",
    "hotmails.com", "hotpop.com", "hulapla.de", "humaility.com",
    "ieatspam.eu", "ieatspam.info", "ihateyoualot.info",
    "iheartspam.org", "imailto.net", "imgof.com", "inbax.tk",
    "inbox.si", "inboxclean.com", "inboxclean.org",
    "incognitomail.com", "incognitomail.net", "incognitomail.org",
    "instant-mail.de", "instantemailaddress.com", "iodizc.com",
    "ip6.li", "ipoo.org", "irish2me.com", "jetable.com",
    "jetable.net", "jetable.org", "joseiaspe.com",
    "kasmail.com", "kaspop.com", "killmail.com", "killmail.net",
    "klzlk.com", "knol-power.nl", "kostenlosemailadresse.de",
    "kurzepost.de", "l33r.eu", "lackmail.ru", "lags.us",
    "landmail.co.uk", "laoeq.com", "lastmail.co", "leafmailer.com",
    "letthemeatspam.com", "lhsdv.com", "lifebyfood.com",
    "link2mail.net", "litedrop.com", "lolfreak.net", "lookugly.com",
    "lortemail.dk", "losemymail.com", "lovemeleaveme.com",
    "lr78.com", "lukop.dk", "m4ilweb.info", "macr2.com", "mail114.net",
    "mail1a.de", "mail21.cc", "mail2rss.org", "mail333.com",
    "mail4trash.com", "mail707.com", "mail72.com", "mail7.io",
    "mailbidon.com", "mailbiz.biz", "mailblocks.com", "mailblog.biz",
    "mailbucket.org", "mailcat.biz", "mailcatch.com", "mailde.de",
    "mailde.info", "mailexpire.com", "mailf5.com",
    "mailforspam.com", "mailfreeonline.com", "mailfs.com",
    "mailguard.me", "mailhazard.com", "mailhazard.us",
    "mailimate.com", "mailin8r.com", "mailinater.com",
    "mailink.net", "mailme.ir", "mailme.lv", "mailme24.com",
    "mailmetrash.com", "mailmoat.com", "mailnew.com",
    "mailnull.com", "mailpick.biz", "mailproxsy.com",
    "mailquack.com", "mailrock.biz", "mailsac.com",
    "mailscrap.com", "mailshell.com", "mailsiphon.com",
    "mailslite.com", "mailsucker.net", "mailtome.de", "mailtothis.com",
    "mailtrash.net", "mailtv.net", "mailtv.tv", "mailzilla.com",
    "mailzilla.org", "makemetheking.com", "malahov.de",
    "manifestgenerator.com", "manybrain.com", "mbx.cc",
    "mega.zik.dj", "megaleak.net", "meinspamschutz.de", "melfki.com",
    "mezimages.net", "mfsa.ru", "mierdamail.com", "migumail.com",
    "minimail.eu", "ministry-of-silly-walks.de", "mintemail.com",
    "misterpinball.de", "moburl.com", "mockmail.com", "moeri.org",
    "momentics.ru", "monmail.fr.nf", "mox.pp.ua", "mt2009.com",
    "mt2014.com", "mx0.wwwnew.eu", "myalias.pw",
    "mymail-in.net", "mypacks.net", "mypartyclip.de",
    "myphantomemail.com", "mysamp.de", "mytempemail.com",
    "mytrashmail.com", "netzidiot.de", "neutralspam.com",
    "newbpotato.tk", "nfast.net", "niftynitwit.com",
    "nincsmail.com", "nnh.com", "no-spam.ws", "noblepioneer.com",
    "nobulk.com", "nodezero.net", "nomail.pw",
    "nomail2me.com", "nomorespamemails.com", "notmailinator.com",
    "nospamthanks.info", "notrnailinator.com", "nowhere.org",
    "nowmymail.com", "ntlhelp.net", "nullmail.com",
    "nwldx.com", "objectmail.com", "obobbo.com",
    "odnorazovoe.ru", "one-time.email", "oneoffemail.com",
    "onewaymail.com", "onlatedotcom.info", "online.ms",
    "onqin.com", "oopi.org", "opentrash.com", "ordinaryamerican.net",
    "otherinbox.com", "ourpreviewdomain.com", "owlpic.com",
    "pancakemail.com", "paplease.com", "pepbot.com",
    "phentermine-mortgages.com", "pimpedupmyride.com",
    "pjjkp.com", "pookmail.com", "postalmail.biz",
    "privacy.net", "privymail.de", "proxymail.eu",
    "prtnx.com", "prtz.eu", "pubmail.io",
    "putthisinyourspamdatabase.com", "pwrby.com",
    "quickinbox.com", "quickmail.in", "quickmail.nl",
    "recode.me", "recursor.net", "recyclemail.dk",
    "regbypass.com", "regbypass.comsafe-mail.net",
    "rejectmail.com", "reliable-mail.com", "replyyes.com",
    "rfc822.org", "rhyta.com", "rklips.com",
    "rmqkr.net", "rppkn.com", "rsvhr.com",
    "rudymail.ml", "runbox.com", "safetymail.info",
    "safetypost.de", "sandelf.de", "sanfinder.com",
    "sanstr.com", "sast.ro", "scatmail.com",
    "secretemail.de", "secure-email.com", "sendspamhere.com",
    "senseless-entertainment.com", "services391.com",
    "sharedmailbox.org", "sharklasers.com",
    "shieldedmail.com", "shiftmail.com", "shitmail.de",
    "shitmail.me", "shitmail.org", "shitware.nl",
    "shmeriously.com", "shortmail.net",
    "sibmail.com", "sinnlos-mail.de", "sino.tw",
    "skeefmail.com", "slapsfromlastnight.com",
    "slaskpost.se", "slipry.net", "slopsbox.com",
    "slothmail.net", "slushmail.com", "smellfear.com",
    "smellrear.com", "snakemail.com", "sneakemail.com",
    "snkmail.com", "sofimail.com", "sofort-mail.de",
    "sogetthis.com", "sohu.com", "soodonims.com",
    "spam.su", "spam4.me", "spamavert.com",
    "spambob.com", "spambob.net", "spambob.org",
    "spambog.com", "spambog.de", "spambog.ru",
    "spamcon.org", "spamcorptastic.com", "spamcowboy.com",
    "spamcowboy.net", "spamcowboy.org", "spamday.com",
    "spamex.com", "spamfree.eu", "spamfree24.de",
    "spamfree24.eu", "spamfree24.info", "spamfree24.net",
    "spamfree24.org", "spamgoes.in", "spamgourmet.com",
    "spamgourmet.net", "spamgourmet.org", "spamherelots.com",
    "spamhereplease.com", "spamhole.com", "spamify.com",
    "spaminator.de", "spamkill.info", "spaml.com",
    "spaml.de", "spammotel.com", "spammy.com", "spamoff.de",
    "spamslicer.com", "spamspot.com", "spamstack.net",
    "spamthis.co.uk", "spamthisplease.com", "spamtroll.net",
    "speed.1s.fr", "spl0it.com", "spoofmail.de",
    "squizzy.de", "squizzy.eu", "squizzy.net",
    "ssoia.com", "startfu.com", "stevemail.pl", "stinkefinger.net",
    "stop-my-spam.com", "stuffmail.de", "supergreatmail.com",
    "supermailer.jp", "superrito.com", "suremail.info",
    "svk.jp", "sweetxxx.de", "tafmail.com", "tagyourself.com",
    "taylorventuresllc.com", "tefl.ro", "telecomix.pl",
    "teleworm.us", "teml.net", "temp-mail.com",
    "temp-mail.de", "temp.bartdevos.be", "tempail.com",
    "tempalias.com", "tempe-mail.com", "tempemail.biz",
    "tempemail.co.za", "tempemail.com", "tempemail.net",
    "tempinbox.co.uk", "tempinbox.com", "tempmail.de",
    "tempmail.eu", "tempmail.it", "tempmail.us",
    "tempmail2.com", "tempmaildemo.com", "tempmailer.com",
    "tempmailer.de", "tempomail.fr", "temporaryemail.net",
    "temporaryemail.us", "temporaryforwarding.com",
    "temporaryinbox.com", "temporarymailaddress.com",
    "tempsky.com", "tempthe.net", "tempymail.com",
    "thanksnospam.info", "thecloudindex.com",
    "thenullemail.com", "thisisnotmyrealemail.com",
    "throam.com", "throwam.com", "throwam.net",
    "throwaway.email", "throwaymail.com", "tilien.com",
    "tittbit.in", "tmailinator.com", "toiea.com",
    "tradermail.info", "trash-amil.com", "trash-mail.at",
    "trash-mail.cf", "trash-mail.com", "trash-mail.de",
    "trash-mail.ga", "trash-mail.gq", "trash-mail.io",
    "trash-mail.ml", "trash-mail.tk", "trashemail.de",
    "trashimail.de", "trashmail.at", "trashmail.com",
    "trashmail.io", "trashmail.me", "trashmail.net",
    "trashmail.org", "trashmail.xyz", "trashmailer.com",
    "trashspam.com", "trayna.com", "trbvm.com",
    "treemail.de", "trollproject.com", "tropicalbass.info",
    "trout.addressfinder.net", "trv6.com", "tunxis.info",
    "turual.com", "twinmail.de", "tyldd.com",
    "uggsrock.com", "umail.net", "unmail.ru",
    "upliftnow.com", "uplipht.com", "uroid.com",
    "us.af", "uyhip.com", "venompen.com",
    "veryrealemail.com", "viditag.com", "viewcastmedia.com",
    "viewcastmedia.net", "viewcastmedia.org",
    "viralplays.com", "vkcode.ru", "vomoto.com",
    "votiputox.org", "vpn.st", "vsimcard.com",
    "vubby.com", "walala.org", "walkmail.net",
    "walkmail.ru", "webemail.me", "weg-werf-email.de",
    "weggam.com", "wegwerf-email.at", "wegwerf-email.de",
    "wegwerf-email.net", "wegwerf-email.org",
    "wegwerfadresse.de", "wegwerfmail.de", "wegwerfmail.net",
    "wegwerfmail.org", "wilemail.com", "willhackforfood.biz",
    "willselfdestruct.com", "winemaven.info",
    "wronghead.com", "wuzup.net", "wuzupmail.net",
    "www.e4ward.com", "www.mailinator.com", "wwwnew.eu",
    "xagloo.com", "xemaps.com", "xents.com", "xmaily.com",
    "xoxy.net", "xwaretech.com", "xwaretech.info",
    "xwaretech.net", "xww.ro", "xyz.am",
    "y7mail.com", "yahomail.top", "yapped.net",
    "yeah.net", "yep.it", "yogamaven.com",
    "yopmail.com", "yopmail.fr", "yopmail.gq",
    "yopmail.net", "youmail.ga", "yourdomain.com",
    "yourspamgoeshere.com", "ypmail.webarnak.fr.eu.org",
    "yuurok.com", "z1p.biz", "za.com", "zehnminuten.de",
    "zehnminutenmail.de", "zetmail.com", "zippymail.info",
    "zoaxe.com", "zoemail.com", "zoemail.net",
    "zoemail.org", "zomg.info", "zxcv.com", "zxcvbnm.com",
    "zzz.com",
    # --- extra burner/alias services ---
    "mailsac.com", "spamgourmet.com", "notmailinator.com",
    "spamoff.de", "harakirimail.com", "spam.la", "trashmail.com",
    "discardmail.com", "mailnull.com", "safetypost.de",
    "throwam.com", "binkmail.com", "bobmail.info", "chammy.info",
    "devnullmail.com", "divad.ga", "dudmail.com", "durandinterstellar.com",
    "easytrashmail.com", "ephemail.net", "explodemail.com",
    "extra.com", "eyepaste.com", "fismail.com", "freemail.ms",
    "ftpinc.ca", "friscaa.com", "gemq.com",
    "haltospam.com", "hezll.com", "hidebox.org", "hidemail.de",
    "hinote.com", "hostguru.info", "hotpop.com", "iheartspam.org",
    "inoutmail.de", "inoutmail.eu", "inoutmail.info", "inoutmail.net",
    "internet-e-mail.de", "internet-mail.de", "internetemails.net",
    "internetemails.org", "junk1.tk", "kasmail.com", "killmail.com",
    "knol-power.nl", "lackmail.ru", "lacto.com",
    "link2mail.net", "littleapple.com", "litedrop.com",
    "lol.ovpn.to", "lookugly.com",
    "maildx.com", "mailguard.me", "mailme.lv",
    "mailnew.com", "mailquack.com", "mailrock.biz",
    "mailshell.com", "mailtv.tv", "mailzilla.com",
    "mbx.cc", "mierdamail.com", "mintemail.com",
    "mycleaninbox.net", "mypartyclip.de",
    "nezdiro.org", "noblepioneer.com",
    "noclickemail.com", "nospam.ze.tc", "nowmymail.com",
    "notsharingmy.info", "objectmail.com",
    "one-time.email", "onewaymail.com",
    "pookmail.com", "postalmail.biz", "privacy.net",
    "proxymail.eu", "rejectmail.com",
    "safetymail.info", "scatmail.com",
    "spoofmail.de", "stop-my-spam.com",
    "supermailer.jp", "svk.jp",
    "tempalias.com", "tempe-mail.com",
    "tempmaildemo.com", "tempsky.com",
    "throwaway.email", "tilien.com",
    "trout.addressfinder.net", "tunxis.info",
    "uggsrock.com", "upliftnow.com", "uroid.com",
    "walkmail.ru", "yuurok.com",
    "zehnminutenmail.de", "zoaxe.com",
})

# ---------------------------------------------------------------------------
# Top-50 most commonly used / easily guessable passwords to block outright.
# ---------------------------------------------------------------------------
_COMMON_PASSWORDS: frozenset[str] = frozenset({
    "password", "password1", "password123", "password1234", "password12345",
    "123456789", "1234567890", "12345678901", "123456789012",
    "qwerty123", "qwerty1234", "qwertyuiop", "letmein", "letmein1",
    "iloveyou", "iloveyou1", "monkey123", "dragon123",
    "master123", "abc123456", "trustno1", "admin1234",
    "welcome1", "welcome123", "football1", "baseball1",
    "sunshine1", "shadow123", "superman1", "batman1234",
    "mustang1", "michael1", "jessica1", "charlie1",
    "jordan123", "harley123", "ranger123",
    "passw0rd", "p@ssword", "p@ssw0rd", "p@55word",
    "admin@123", "admin1234", "root1234",
    "qwerty!123", "qwerty@123",
    "changeme1", "changeme!", "changeme123",
})


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=14, max_length=128)

    @field_validator("email")
    @classmethod
    def email_not_disposable(cls, value: str) -> str:
        domain = value.split("@", 1)[-1].lower().strip()
        if domain in _DISPOSABLE_DOMAINS:
            raise ValueError(
                "Disposable or temporary email addresses are not permitted. "
                "Please use a valid institutional or personal email provider."
            )
        if "." not in domain:
            raise ValueError("Email domain must be a fully qualified domain name.")
        return value.lower().strip()

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        errors: list[str] = []
        if not re.search(r"[A-Z]", value):
            errors.append("at least one uppercase letter (A-Z)")
        if not re.search(r"[a-z]", value):
            errors.append("at least one lowercase letter (a-z)")
        if not re.search(r"\d", value):
            errors.append("at least one digit (0-9)")
        if not re.search(r"[!@#$%^&*()\-_=+\[\]{};:'\",.<>?/\\|`~]", value):
            errors.append("at least one special character (!@#$%^&* etc.)")
        if re.search(r"(.)\1{3,}", value):
            errors.append("no more than 3 consecutive identical characters")
        if errors:
            raise ValueError("Password must contain: " + "; ".join(errors))
        if value.lower() in _COMMON_PASSWORDS:
            raise ValueError(
                "This password is too common and easily guessable. "
                "Please choose a more unique password."
            )
        return value

    @model_validator(mode="after")
    def password_not_contain_email(self) -> "RegisterRequest":
        """Prevent passwords that embed the user's own email local-part."""
        email_local = self.email.split("@")[0].lower()
        if len(email_local) >= 4 and email_local in self.password.lower():
            raise ValueError(
                "Password must not contain your email address or username."
            )
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class HandshakeRequest(BaseModel):
    x25519_public_key: str
    ml_kem_public_key: str | None = None
    client_nonce: str


class OrderRequest(BaseModel):
    asset: str = Field(min_length=1, max_length=20)  # raised to 20 to accommodate e.g. RELIANCE.NS
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0, le=1_000_000)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    time_in_force: Literal["day", "gtc", "ioc"] = "day"

    @field_validator("asset")
    @classmethod
    def normalized_asset(cls, value: str) -> str:
        value = value.strip().upper()
        # Allow A-Z, digits, dots (e.g. RELIANCE.NS, 7203.T), hyphens (BTC-USD)
        if not re.fullmatch(r"[A-Z0-9.\-]{1,20}", value):
            raise ValueError("asset must be a valid ticker symbol (letters, digits, '.', '-')")
        return value


class RotateKeysRequest(BaseModel):
    algorithm: str = "ML-DSA-65"
    reason: str = "scheduled_rotation"


class StrategyRequest(BaseModel):
    name: str = Field(min_length=3, max_length=80)
    asset: str = Field(min_length=1, max_length=20)
    fast_window: int = Field(default=20, ge=2, le=100)
    slow_window: int = Field(default=50, ge=5, le=250)

    @field_validator("asset")
    @classmethod
    def normalized_strategy_asset(cls, value: str) -> str:
        return OrderRequest.normalized_asset(value)

    @field_validator("slow_window")
    @classmethod
    def valid_windows(cls, value: int, info) -> int:
        if "fast_window" in info.data and value <= info.data["fast_window"]:
            raise ValueError("slow_window must be larger than fast_window")
        return value


class BacktestRequest(BaseModel):
    asset: str = Field(min_length=1, max_length=20)
    fast_window: int = Field(default=20, ge=2, le=100)
    slow_window: int = Field(default=50, ge=5, le=250)
    period: str = "1y"

    @field_validator("asset")
    @classmethod
    def normalized_backtest_asset(cls, value: str) -> str:
        return OrderRequest.normalized_asset(value)

    @field_validator("period")
    @classmethod
    def valid_period(cls, value: str) -> str:
        if value not in {"6mo", "1y", "2y", "5y"}:
            raise ValueError("period must be one of 6mo, 1y, 2y, 5y")
        return value


class ApiKeyRequest(BaseModel):
    name: str = Field(min_length=3, max_length=80)
    scopes: list[str] = Field(default_factory=lambda: ["read"])

    @field_validator("scopes")
    @classmethod
    def valid_scopes(cls, value: list[str]) -> list[str]:
        allowed = {"read", "trade", "admin"}
        scopes = sorted(set(value))
        if not scopes or not set(scopes).issubset(allowed):
            raise ValueError("scopes must be read, trade, and/or admin")
        return scopes


class WebhookRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    event_types: list[str] = Field(default_factory=lambda: ["order.filled", "key.rotated"])

    @field_validator("url")
    @classmethod
    def safe_webhook_url(cls, value: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("webhook URL must be an absolute HTTPS URL without credentials")
        return value

    @field_validator("event_types")
    @classmethod
    def valid_event_types(cls, value: list[str]) -> list[str]:
        allowed = {"order.filled", "order.rejected", "order.cancelled", "key.rotated"}
        events = sorted(set(value))
        if not events or not set(events).issubset(allowed):
            raise ValueError("unsupported webhook event type")
        return events
